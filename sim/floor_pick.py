"""Whole-body G1 floor pick with a contact-only gripper in headless MuJoCo."""

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager, nullcontext
from pathlib import Path

import hydra
import mujoco as mj
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from reach import BIAS_M, WARMUP
from scalebfm_playback import (
    HeadlessSimulator,
    bridge,
    load_policy,
    metadata_path,
    output,
    policy_observation_keys,
)
from scalebridge.env.motion_tracking import MotionTrackingEnv
from scalebridge.simulator.base_simulator import BaseSimulator

WRIST = 'right_wrist_yaw_link'
ARM = [
    f'right_{joint}_joint' for joint in
    ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
]
APPLE_RADIUS = 0.034
APPLE_REST = np.array([0.14, -0.24, APPLE_RADIUS])
PAD_REACH = 0.146
HOLD = 25
CLOSE = 25
STAND = 30
DOWN = 118
REACH = 54
LIFT = 100
RETRACT = 40
FIRST_HOLD = STAND + DOWN
FIRST_CLOSE = FIRST_HOLD + HOLD
FIRST_LIFT = FIRST_CLOSE + CLOSE
LIFT_HEIGHT_M = 0.10
LIFT_CONFIRM_TICKS = 10
RESIDUAL_LIMIT_M = 0.10
EXPLORATION_M = 0.055
# Shared reference calibration for the arm and gripper on this scripted squat.
REFERENCE_OFFSET_M = np.array([-.055, .015, -.130])
# Contact surfaces are normal to local Y, with local X pointing down at pickup.
GRASP = Rotation.from_euler('y', 90, degrees=True)


def vector(values):
    return ' '.join(str(value) for value in values)


class PickSimulator(HeadlessSimulator):
    def _setup_backbone(self):
        BaseSimulator._setup_backbone(self)
        self.has_object = False  # Keep object state out of the tracker's observations.
        source = Path(self.cfg.asset.xml_path).resolve()
        root = ET.parse(source).getroot()
        compiler = root.find('compiler')
        compiler.set('meshdir', str(source.parent / compiler.get('meshdir')))
        wrist = root.find(f".//body[@name='{WRIST}']")
        for geom in list(wrist.findall("geom[@mesh='right_rubber_hand']")):
            wrist.remove(geom)
        ET.SubElement(wrist, 'site', name='pinch', pos=f'{PAD_REACH} 0 0', size='0.006')
        ET.SubElement(
            wrist, 'geom', name='palm', type='box', size='0.016 0.069 0.018',
            pos='0.051 0 0', mass='0.06', contype='0', conaffinity='0', rgba='0.18 0.21 0.25 1',
        )
        for side, sign in (('left', 1), ('right', -1)):
            finger = ET.SubElement(wrist, 'body', name=f'{side}_pad', pos=f'{PAD_REACH} {sign * .069} 0')
            ET.SubElement(
                finger, 'joint', name=f'{side}_slide', type='slide', axis=f'0 {-sign} 0',
                range='0 .050', damping='2.0', armature='0.015', actuatorfrclimited='false',
            )
            ET.SubElement(
                finger, 'geom', name=f'{side}_pad_geom', type='box', size='.029 .008 .037',
                mass='0.035', contype='4', conaffinity='2', friction='2.5 .04 .01',
                condim='6', solref='.012 1', rgba='.25 .30 .35 1',
            )
            ET.SubElement(
                finger, 'geom', name=f'{side}_lip', type='capsule', size='.008',
                fromto=f'.027 {-sign * .009} -.033 .027 {-sign * .009} .033',
                mass='.009', contype='4', conaffinity='2', friction='2.5 .04 .01',
                condim='6', solref='.012 1', rgba='.25 .30 .35 1',
            )
            ET.SubElement(
                finger, 'geom', type='capsule', fromto='-.083 0 0 -.016 0 0',
                size='.008', mass='0.012', contype='0', conaffinity='0',
                rgba='.22 .27 .33 1',
            )
        floor = root.find(".//geom[@name='floor']")
        floor.set('conaffinity', '7')
        apple = ET.SubElement(root.findall('worldbody')[-1], 'body', name='apple', pos=vector(APPLE_REST))
        ET.SubElement(apple, 'freejoint', name='apple_free')
        ET.SubElement(
            apple, 'geom', name='apple_geom', type='sphere', size=str(APPLE_RADIUS),
            mass='.085', contype='2', conaffinity='0', friction='1 .015 .002', rgba='.80 .06 .08 1',
        )
        rolling = self.cfg.get('apple_rolling_resistance', 0)
        if rolling:
            contacts = root.find('contact')
            if contacts is None:
                contacts = ET.SubElement(root, 'contact')
            ET.SubElement(
                contacts, 'pair', geom1='floor', geom2='apple_geom', condim='6',
                friction=vector([1., 1., .015, rolling, rolling]),
            )
        for side in ('left', 'right'):
            ET.SubElement(
                root.find('actuator'), 'position', name=f'{side}_servo', joint=f'{side}_slide',
                kp='850', ctrllimited='true', ctrlrange='0 .05', forcelimited='true', forcerange='-25 25',
            )
        self.mujoco_model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.mujoco_model.opt.timestep = self.low_dt
        self.mujoco_data = mj.MjData(self.mujoco_model)
        self.robot_actuators = np.array(
            [i for i in range(self.mujoco_model.nu) if not self.mujoco_model.actuator(i).name.endswith('_servo')]
        )
        self.finger_actuators = [self.mujoco_model.actuator(f'{side}_servo').id for side in ('left', 'right')]
        self.apple_geom = self.mujoco_model.geom('apple_geom').id
        self.pad_sides = {
            self.mujoco_model.geom(f'{side}_{name}').id: side
            for side in ('left', 'right') for name in ('pad_geom', 'lip')
        }
        self.apple_address = int(self.mujoco_model.joint('apple_free').qposadr[0])
        self.pinching = False
        self.bilateral_contact = False

    def _get_joint_names(self):
        return [self.mujoco_model.actuator(i).name for i in self.robot_actuators]

    def apply_action(self, tgt_dof_pos):
        target = self.default_dof_pos.copy()
        target[self.env_action_to_sim_idx] = tgt_dof_pos.squeeze()
        data = self.mujoco_data
        self.bilateral_contact = False
        for _ in range(self.decimation):
            data.ctrl[self.robot_actuators] = (
                (target - data.qpos[7:7 + self.num_joints]) * self.stiffness
                - data.qvel[6:6 + self.num_joints] * self.damping
            )
            data.ctrl[self.finger_actuators] = 0.050 if self.pinching else 0
            mj.mj_step(self.mujoco_model, data)
            contacted = set()
            for contact in data.contact[:data.ncon]:
                pair = {contact.geom1, contact.geom2}
                if self.apple_geom in pair:
                    contacted.update(self.pad_sides[geom] for geom in pair if geom in self.pad_sides)
            self.bilateral_contact |= len(contacted) == 2
        mj.mj_fwdPosition(self.mujoco_model, data)

    def contact_point(self):
        return self.mujoco_data.site('pinch').xpos.copy()

    def apple_position(self):
        return self.mujoco_data.qpos[self.apple_address:self.apple_address + 3].copy()


class PickEnv(MotionTrackingEnv):
    def _calibrate(self):
        self.simulator.calibrate()
        self._gather_reference_state()


def build_reference(sim, metadata, motion_path, target):
    """Keep the squat's body motion and edit its right arm with wrist-pose IK."""
    source = np.load(motion_path)
    model = sim.mujoco_model
    data = mj.MjData(model)
    names = metadata['joint_names']
    joint_qpos = np.array([int(model.joint(name).qposadr[0]) for name in names])
    arm = [names.index(name) for name in ARM]
    bounds = np.array([model.joint(name).range for name in ARM])
    base_pos = source['body_pos_w'][0, 0].copy()
    base_pos[2] = 0
    root_rotation = Rotation.from_quat(source['body_quat_w'][0, 0], scalar_first=True)
    initial_forward = root_rotation.as_matrix()[:, 0]
    heading = np.arctan2(initial_forward[1], initial_forward[0])
    align = Rotation.from_rotvec([0, 0, -heading])
    down_frames = np.rint(np.linspace(0, 60, DOWN)).astype(int)
    lift_frames = np.rint(np.linspace(60, 0, LIFT)).astype(int)
    frames = np.r_[np.zeros(STAND, dtype=int), down_frames, np.full(HOLD + CLOSE, 60), lift_frames,
                   np.zeros(RETRACT, dtype=int)]
    body_ids = [model.body(name).id for name in metadata['selected_body_names']]
    positions, rotations, nominal_points = [], [], []
    first_reach = FIRST_HOLD - REACH
    wrist = model.body(WRIST).id
    nominal_grasp = None
    previous = None
    for tick, frame in enumerate(frames):
        data.qpos[joint_qpos] = source['joint_pos'][frame]
        data.qpos[:3] = align.apply(source['body_pos_w'][frame, 0] - base_pos)
        data.qpos[3:7] = (
            align * Rotation.from_quat(source['body_quat_w'][frame, 0], scalar_first=True)
        ).as_quat(scalar_first=True)
        mj.mj_fwdPosition(model, data)
        if first_reach <= tick < FIRST_LIFT:
            progress = min(1., (tick - first_reach) / (REACH - 1))
            progress = progress**2 * (3 - 2 * progress)
            natural = Rotation.from_quat(data.xquat[wrist], scalar_first=True)
            wanted_rotation = natural * Rotation.from_rotvec((natural.inv() * GRASP).as_rotvec() * progress)
            natural_point = data.site('pinch').xpos.copy()
            goal = natural_point * (1 - progress) + target * progress
            wanted_wrist = goal - wanted_rotation.apply([PAD_REACH, 0, 0])
            if previous is None:
                previous = data.qpos[joint_qpos[arm]].copy()

            def error(joints):
                data.qpos[joint_qpos[arm]] = joints
                mj.mj_fwdPosition(model, data)
                current = Rotation.from_quat(data.xquat[wrist], scalar_first=True)
                return np.r_[data.xpos[wrist] - wanted_wrist,
                             .14 * (wanted_rotation.inv() * current).as_rotvec(),
                             .004 * (joints - previous)]

            previous = least_squares(error, np.clip(previous, *bounds.T), bounds=tuple(bounds.T),
                                     max_nfev=40, ftol=1e-6).x
            data.qpos[joint_qpos[arm]] = previous
            nominal_grasp = previous
        elif tick >= FIRST_LIFT:
            # Keep the arm in the grasp pose while the squat reverses, then tuck it.
            progress = np.clip((tick - FIRST_LIFT - 50) / 48, 0, 1)
            progress = progress**2 * (3 - 2 * progress)
            data.qpos[joint_qpos[arm]] = nominal_grasp * (1 - progress) + data.qpos[joint_qpos[arm]] * progress
        mj.mj_fwdPosition(model, data)
        frame_positions = data.xpos[body_ids].copy()
        frame_rotations = data.xquat[body_ids].copy()
        if first_reach <= tick < FIRST_LIFT:
            # The command encodes the intended pinch point even at the IK workspace boundary.
            frame_positions[-1] = goal - wanted_rotation.apply([PAD_REACH, 0, 0])
            frame_rotations[-1] = wanted_rotation.as_quat(scalar_first=True)
        positions.append(frame_positions)
        rotations.append(frame_rotations)
        nominal_points.append(goal if first_reach <= tick < FIRST_LIFT else data.site('pinch').xpos.copy())
    return np.array(positions), np.array(rotations), np.array(nominal_points)


def set_reference(env, reference):
    env.body_pos_w = torch.as_tensor(reference[0], dtype=torch.float32, device=env.device)
    env.body_quat_w = torch.as_tensor(reference[1], dtype=torch.float32, device=env.device)
    env.motion_len = len(reference[0])


class HoldModel:
    """Learn the local XY response of the whole-body tracker from completed holds."""

    def __init__(self):
        self.commands = []
        self.offsets = []

    def observe(self, correction_xy, hand_xy, target_xy):
        self.commands.append(np.r_[correction_xy, 1.0])
        self.offsets.append(hand_xy - target_xy)

    def correction(self):
        design = np.array(self.commands)
        offsets = np.array(self.offsets)
        regularization = np.diag([0.0003, 0.0003, 0.0])
        weights = np.linalg.solve(design.T @ design + regularization, design.T @ offsets)
        jacobian = weights[:2].T
        correction = np.linalg.solve(
            jacobian.T @ jacobian + 0.02 * np.eye(2), -jacobian.T @ weights[2],
        )
        return np.clip(correction, -RESIDUAL_LIMIT_M, RESIDUAL_LIMIT_M)


@contextmanager
def record_video(sim, path):
    import imageio.v2 as imageio

    camera = mj.MjvCamera()
    camera.lookat[:] = [0, -.08, .55]
    camera.distance = 1.9
    camera.azimuth = 122
    camera.elevation = -15
    path.parent.mkdir(parents=True, exist_ok=True)
    with mj.Renderer(sim.mujoco_model, height=480, width=640) as renderer:
        with imageio.get_writer(path, fps=25, codec='libx264') as writer:
            def append_frame():
                renderer.update_scene(sim.mujoco_data, camera=camera)
                writer.append_data(renderer.render())

            yield append_frame


def reset_episode(env, target):
    sim = env.simulator
    sim.pinching = False
    sim.default_qpos[sim.apple_address:sim.apple_address + 3] = target
    mj.mj_resetData(sim.mujoco_model, sim.mujoco_data)
    env.reset()


def play_attempt(
    env, policy, target, bias, correction_xy, append_frame=None, recover=False,
    skip_close=False,
):
    """Play a pick from live physics state. Only the motion clock starts over."""
    sim = env.simulator
    env.episode_length_buf.zero_()
    env._gather_reference_state()
    mode = torch.tensor([7], device=env.device)
    residual = np.r_[correction_xy, 0.]
    future_offsets = env.future_frame_offset.cpu().numpy()
    release = False
    missed_contact = 0
    fall_ticks = 0
    lift_height = target[2] + LIFT_HEIGHT_M
    trace = []
    for tick in range(env.motion_len):
        # Ramp the wrist command offset in during reach and out over initial lift.
        future = np.clip(tick + future_offsets, 0, env.motion_len - 1)
        blend_in = np.clip((future - (FIRST_HOLD - REACH)) / 25, 0, 1)
        blend_out = np.clip(1 - (future - FIRST_LIFT) / 22, 0, 1)
        factor = blend_in * blend_out
        command_offset = torch.as_tensor(
            factor[:, None] * (REFERENCE_OFFSET_M + bias + residual),
            dtype=torch.float32, device=env.device,
        )
        env.state_buffer['body_pos_w_future'][0, :, -1] += command_offset
        obs = env._update_observation_manager()
        qpos, action = policy(*(obs[key] for key in policy_observation_keys), mode, obs['future_time_offsets'])
        if not torch.isfinite(action).all():
            raise RuntimeError(f'Nonfinite tracker action at frame {tick}')
        if recover and tick >= FIRST_LIFT:
            missed_contact = 0 if sim.bilateral_contact else missed_contact + 1
            # Treat lost contact or a grasp still at floor height late in the rise as a miss.
            release |= missed_contact >= 5 or (
                tick == FIRST_LIFT + 70 and sim.apple_position()[2] < lift_height
            )
        sim.pinching = tick >= FIRST_CLOSE and not (release or skip_close)
        env.step(qpos, action)
        pinch = sim.contact_point()
        apple = sim.apple_position()
        trace.append(np.r_[pinch, apple, sim.mujoco_data.qpos[2], int(sim.bilateral_contact),
                          factor[0] * residual, sim.mujoco_data.body(WRIST).xquat, sim.mujoco_data.qpos[:2]])
        if append_frame is not None and tick % 2 == 0:
            append_frame()
        if recover:
            root = sim.mujoco_data
            fallen = root.qpos[2] < 0.15 or root.body('pelvis').xmat[8] < 0.42
            fall_ticks = fall_ticks + 1 if fallen else 0
            if fall_ticks >= 5:
                break
    trace = np.array(trace)
    hold = trace[FIRST_HOLD:FIRST_CLOSE, :3]
    hold_error = np.linalg.norm(hold - target, axis=1)
    # Require a sustained lift with bilateral contact. A tossed apple cannot satisfy this.
    lifted = (trace[FIRST_LIFT:, 5] >= lift_height) & (trace[FIRST_LIFT:, 7] == 1)
    longest = max((len(run) for run in ''.join(str(int(hit)) for hit in lifted).split('0')), default=0)
    metrics = {
        'hold_median_error_cm': 100 * np.median(hold_error) if len(hold) else None,
        'hold_median_xy_error_cm': (
            100 * np.median(np.linalg.norm(hold[:, :2] - target[:2], axis=1)) if len(hold) else None
        ),
        'peak_apple_rise_cm': max(0., 100 * (np.max(trace[FIRST_LIFT:, 5], initial=target[2]) - target[2])),
        'lift_frames': longest,
        'lifted': int(longest >= LIFT_CONFIRM_TICKS),
    }
    if recover:
        metrics['fell'] = int(fall_ticks >= 5)
        metrics['released_on_miss'] = int(release)
        metrics['lifted'] &= int(len(lifted) > 0 and lifted[-1])
    hand_xy = np.median(hold[-10:, :2], axis=0) if len(hold) == HOLD else None
    return metrics, hand_xy, trace


def attempt(env, policy, target, bias, correction_xy, append_frame=None):
    reset_episode(env, target)
    return play_attempt(env, policy, target, bias, correction_xy, append_frame)


def make_environment(apple_rolling_resistance=0.0):
    metadata = json.loads(metadata_path.read_text())
    os.chdir(bridge)
    with hydra.initialize_config_dir(config_dir=str(bridge / 'scalebridge/config'), version_base=None):
        cfg = hydra.compose(config_name='base', overrides=[
            f'simulator._target_={PickSimulator.__module__}.PickSimulator',
            f'env._target_={PickEnv.__module__}.PickEnv',
            'env.config.reference_forcing=false',
            f'+simulator.config.apple_rolling_resistance={apple_rolling_resistance}',
        ])
    return instantiate(cfg.env, metadata_dict=metadata), metadata, Path(cfg.env.config.motion_path)


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--probe', action='store_true', help='Try one nominal run without injected bias')
    parser.add_argument('--apple-xy', type=float, nargs=2, default=APPLE_REST[:2])
    parser.add_argument('--output', type=Path, default=output / 'scalebfm_floor_pick.csv')
    parser.add_argument('--video', type=Path, help='Record one attempt as an MP4')
    parser.add_argument('--video-attempt', type=int, default=1)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('--attempts must be positive')
    if args.video and not 1 <= args.video_attempt <= (1 if args.probe else args.attempts):
        parser.error('--video-attempt must refer to an attempt in this run')
    if not np.isfinite(args.apple_xy).all():
        parser.error('--apple-xy must be finite')
    args.output = args.output.resolve()
    if args.video:
        args.video = args.video.resolve()
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    policy = load_policy()
    env, metadata, motion_path = make_environment()
    target = np.r_[args.apple_xy, APPLE_RADIUS]
    reference = build_reference(env.simulator, metadata, motion_path, target)
    set_reference(env, reference)
    learner = HoldModel()
    rng = np.random.default_rng(args.seed)
    warmup = EXPLORATION_M * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    rng.shuffle(warmup)
    rows = []
    with torch.inference_mode():
        for index in range(1, (1 if args.probe else args.attempts) + 1):
            condition = 'nominal' if args.probe else (
                'warmup' if index <= WARMUP else ('control', 'model', 'explore')[(index - WARMUP - 1) % 3]
            )
            correction_xy = np.zeros(2)
            if condition == 'warmup':
                correction_xy = warmup[index - 1]
            elif condition == 'explore':
                correction_xy = rng.uniform(-EXPLORATION_M, EXPLORATION_M, size=2)
            elif condition == 'model':
                correction_xy = learner.correction()
            bias = np.zeros(3) if args.probe else BIAS_M
            recording = (
                record_video(env.simulator, args.video)
                if args.video and index == args.video_attempt else nullcontext()
            )
            with recording as append_frame:
                metrics, hand_xy, trace = attempt(env, policy, target, bias, correction_xy, append_frame)
            learner.observe(correction_xy, hand_xy, target[:2])
            row = {
                'seed': args.seed, 'attempt': index, 'mode': condition,
                'residual_x_m': correction_xy[0], 'residual_y_m': correction_xy[1], **metrics,
            }
            rows.append(row)
            print(f"{index:3d} {condition:8s} hold {metrics['hold_median_error_cm']:6.2f} cm, "
                  f"peak rise {metrics['peak_apple_rise_cm']:6.2f} cm, "
                  f"sustained contact {metrics['lift_frames']} ticks", flush=True)
            if condition == 'nominal' or index in (WARMUP + 1, WARMUP + 2):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    args.output.with_name(f'{args.output.stem}_{condition}_trace.csv'),
                    np.c_[np.arange(len(trace)), trace, reference[2]], delimiter=',', fmt='%.6f',
                    header='tick,hand_x,hand_y,hand_z,apple_x,apple_y,apple_z,pelvis_z,bilateral,'
                           'residual_x,residual_y,residual_z,wrist_w,wrist_x,wrist_y,wrist_z,pelvis_x,pelvis_y,'
                           'nominal_x,nominal_y,nominal_z', comments='',
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.output, rows)


if __name__ == '__main__':
    main()
