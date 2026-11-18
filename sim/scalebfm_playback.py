"""Replay ScaleBFM's bundled G1 squat using eager CUDA and headless MuJoCo."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import hydra
import mujoco
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger

root = Path(__file__).resolve().parents[1]
repo = root / 'resources/ScaleBFM'
bridge = repo / 'ScaleBridge'
models = root / 'resources/ScaleBFM-models'
metadata_path = models / 'model_22200_tensorrt_metadata.json'
output = root / 'results'
policy_observation_keys = (
    'root_quat_buffer', 'base_ang_vel_buffer', 'dof_pos_buffer', 'dof_vel_buffer',
    'actions_buffer', 'target_body_pos_future_to_robot_base',
    'target_body_rot_future_to_robot_base',
)
sys.path[:0] = [str(bridge), str(bridge / 'scalebridge')]
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.simulator.mujoco_simulator import MujocoSimulator


class HeadlessSimulator(MujocoSimulator):
    def _setup_backbone(self):
        BaseSimulator._setup_backbone(self)
        self.has_object = False
        self.mujoco_model = mujoco.MjModel.from_xml_path(self.cfg.asset.xml_path)
        self.mujoco_model.opt.timestep = self.low_dt
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

    def _render(self):
        pass

    def update_marker_pos(self, marker_pos):
        pass


def load_policy():
    # Reuse the upstream deployment wrapper and its checkpoint loading code.
    # Remove its CLI and stop before the TensorRT compilation step.
    exporter = repo / 'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py'
    source = exporter.read_text()
    head, exporter_body = source.split('import time', 1)
    assert 'parser.parse_known_args()' in head
    exporter_body = ('import time' + exporter_body).replace('import torch_tensorrt\n', '')
    exporter_body = exporter_body.replace('weights_only=False', 'weights_only=True')
    core = exporter_body.split('        root_pos = torch.zeros(1, 3', 1)[0]
    assert 'wrapper.eval().cuda()' in core
    with tempfile.TemporaryDirectory(prefix='scalebfm-eager-') as staging:
        adapted = Path(staging) / 'eager_exporter.py'
        adapted.write_text(core + '        return wrapper\n')
        spec = importlib.util.spec_from_file_location('eager_exporter', adapted)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    module.args_cli = SimpleNamespace(
        checkpoint=str(models / 'model_22200.pt'),
        metadata=str(metadata_path),
        mode_table=str(models / 'mode_table.pt'),
        xml_path=str(repo / 'ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml'),
    )
    return module.main()


def main():
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    policy = load_policy()
    os.chdir(bridge)
    with hydra.initialize_config_dir(config_dir=str(bridge / 'scalebridge/config'), version_base=None):
        cfg = hydra.compose(config_name='base', overrides=['simulator._target_=__main__.HeadlessSimulator'])
    env = instantiate(cfg.env, metadata_dict=json.loads(metadata_path.read_text()))
    control_mode = torch.tensor([7], device='cuda')
    heights = []
    joint_errors = []
    with torch.inference_mode():
        obs = env.reset()
        for index in range(env.motion_len):
            targets, action = policy(
                *(obs[key] for key in policy_observation_keys), control_mode, obs['future_time_offsets']
            )
            if not torch.isfinite(action).all():
                raise RuntimeError(f'Nonfinite action on frame {index}')
            obs = env.step(targets, action)
            heights.append(float(env.simulator.mujoco_data.qpos[2]))
            measured = env.simulator.refresh_sim()['dof_pos'].numpy()
            reference_index = min(index + 1, env.motion_len - 1)
            joint_errors.append(float(np.sqrt(np.mean((measured - env.joint_pos[reference_index]) ** 2))))
    torch.cuda.synchronize()
    frames = np.arange(env.motion_len)
    reference_frames = np.minimum(frames + 1, env.motion_len - 1)
    reference_height = env.body_pos_w[:, 0, 2].cpu().numpy()[reference_frames]
    print(f'{env.motion_len} frames, simulated time {env.simulator.mujoco_data.time:.2f} s')
    print(f'Pelvis height (m): simulated min {min(heights):.3f}, max {max(heights):.3f}, final {heights[-1]:.3f}')
    print(f'Pelvis height (m): reference min {min(reference_height):.3f}, max {max(reference_height):.3f}, final {reference_height[-1]:.3f}')
    print(f'Median joint RMSE {np.median(joint_errors):.3f} rad')
    print(f'Peak allocated CUDA memory {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB')
    output.mkdir(exist_ok=True)
    np.savetxt(
        output / 'scalebfm_squat.csv',
        np.column_stack((frames, heights, reference_height, joint_errors)),
        delimiter=',',
        header='frame,pelvis_height_m,reference_height_m,joint_rmse_rad',
        comments='',
        fmt=('%d', '%.6f', '%.6f', '%.6f'),
    )


if __name__ == "__main__":
    main()
