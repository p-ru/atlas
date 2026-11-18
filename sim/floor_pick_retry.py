"""Run paired G1 floor-pick episodes with ground-truth reacquisition and live-state retries."""

import argparse
import hashlib
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from loguru import logger

from floor_pick import (
    APPLE_RADIUS,
    APPLE_REST,
    EXPLORATION_M,
    HoldModel,
    build_reference,
    make_environment,
    play_attempt,
    record_video,
    reset_episode,
    save_csv,
    set_reference,
)
from reach import BIAS_M
from scalebfm_playback import load_policy, output

# The nine matched scenes cover a 7 cm square around the calibrated target.
CASES = {
    'center': (0, 0),
    'forward': (.035, 0),
    'back': (-.035, 0),
    'left': (0, .035),
    'right': (0, -.035),
    'forward-left': (.035, .035),
    'forward-right': (.035, -.035),
    'back-left': (-.035, .035),
    'back-right': (-.035, -.035),
}
STATE_SPEC = mj.mjtState.mjSTATE_INTEGRATION
APPLE_ROLLING_RESISTANCE_M = .003
PICK_WORKSPACE_HALF_WIDTH_M = .14


def state_id(sim):
    model = sim.mujoco_model
    state = np.empty(mj.mj_stateSize(model, STATE_SPEC))
    mj.mj_getState(model, sim.mujoco_data, state, STATE_SPEC)
    return hashlib.sha256(state.tobytes()).hexdigest()


def standing_ready(sim):
    data = sim.mujoco_data
    return (
        data.qpos[2] >= .60 and data.body('pelvis').xmat[8] >= .90
        and np.linalg.norm(data.qvel[:3]) <= .16
        and sim.apple_position()[2] <= APPLE_RADIUS + .015
    )


def warmup(env, policy, metadata, motion_path):
    learner = HoldModel()
    reference = build_reference(env.simulator, metadata, motion_path, APPLE_REST)
    set_reference(env, reference)
    commands = EXPLORATION_M * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    for correction in commands:
        reset_episode(env, APPLE_REST)
        _, measured, _ = play_attempt(env, policy, APPLE_REST, BIAS_M, correction, recover=True)
        if measured is None:
            raise RuntimeError('Warm-up stopped before a complete hold')
        learner.observe(correction, measured, APPLE_REST[:2])
    return learner


def run_episode(
    env, policy, metadata, motion_path, case, mode, max_attempts, learner, video,
    skip_first_close=False, *, frame_hook=None,
):
    sim = env.simulator
    initial = APPLE_REST + np.r_[CASES[case], 0.]
    set_reference(env, build_reference(sim, metadata, motion_path, initial))
    reset_episode(env, initial)
    rows = []
    status = 'budget_exhausted'
    previous_end = None
    recording = record_video(sim, video) if video else nullcontext(frame_hook)
    with recording as append_frame:
        for index in range(1, max_attempts + 1):
            target = sim.apple_position()
            start_time = sim.mujoco_data.time
            start_state = state_id(sim)
            if previous_end is not None:
                if start_state != previous_end:
                    raise RuntimeError('Simulation state changed before the retry')
                reference = build_reference(sim, metadata, motion_path, target)
                if state_id(sim) != start_state:
                    raise RuntimeError('Planning the retry modified live physics state')
                set_reference(env, reference)
            correction = learner.correction() if learner is not None else np.zeros(2)
            interrupted = skip_first_close and index == 1
            metrics, measured, _ = play_attempt(
                env, policy, target, BIAS_M, correction, append_frame=append_frame, recover=True,
                skip_close=interrupted,
            )
            if learner is not None and measured is not None:
                learner.observe(correction, measured, target[:2])
            end = sim.apple_position()
            previous_end = state_id(sim)
            ready = standing_ready(sim)
            if metrics['fell']:
                status = 'fall'
            elif metrics['lifted']:
                status = 'lifted'
            elif index < max_attempts and np.max(np.abs(end[:2] - APPLE_REST[:2])) > PICK_WORKSPACE_HALF_WIDTH_M:
                status = 'outside_pick_workspace'
            elif index < max_attempts and not ready:
                status = 'recovery_unstable'
            rows.append({
                'case': case, 'mode': mode, 'attempt': index,
                'start_time_s': start_time, 'end_time_s': sim.mujoco_data.time,
                'start_state_sha256': start_state, 'end_state_sha256': previous_end,
                **{f'target_{axis}_m': value for axis, value in zip('xyz', target)},
                **{f'end_apple_{axis}_m': value for axis, value in zip('xyz', end)},
                'apple_xy_motion_cm': 100 * np.linalg.norm(end[:2] - target[:2]),
                'correction_x_m': correction[0], 'correction_y_m': correction[1],
                'grasp_interrupted': int(interrupted),
                'ready_to_retry': int(ready),
                **metrics,
            })
            print(
                f'{case:>13} {mode:7} #{index} lift={metrics["lifted"]} fall={metrics["fell"]} '
                f'xy motion={rows[-1]["apple_xy_motion_cm"]:.2f} cm',
                flush=True,
            )
            if status != 'budget_exhausted':
                break
    return {
        'case': case, 'mode': mode, 'initial_x_m': initial[0], 'initial_y_m': initial[1],
        'attempts_used': len(rows), 'first_attempt_lift': int(rows[0]['lifted'] and not rows[0]['fell']),
        'episode_lift': int(status == 'lifted'), 'fell': int(status == 'fall'), 'outcome': status,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, action='append', help='Select a case; default is all nine')
    parser.add_argument('--max-attempts', type=int, default=3, help='Total attempts per episode, including the first')
    parser.add_argument('--output', type=Path, default=output / 'scalebfm_floor_pick_retry.csv')
    parser.add_argument('--video', type=Path, help='Record one complete episode')
    parser.add_argument('--video-case', choices=CASES)
    parser.add_argument('--video-mode', choices=('control', 'learned'), default='learned')
    parser.add_argument(
        '--skip-first-close', action='store_true',
        help='Run a learned-only diagnostic that skips the first gripper closure',
    )
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error('--max-attempts must be positive')
    cases = args.case if args.case is not None else CASES
    if args.video and args.video_case not in cases:
        parser.error('--video requires a --video-case included in the run')
    if args.video and args.skip_first_close and args.video_mode != 'learned':
        parser.error('--skip-first-close runs only the learned condition')
    summary_path = args.output.resolve()
    attempts_path = summary_path.with_name(f'{summary_path.stem}_attempts.csv')
    video = args.video.resolve() if args.video else None
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    policy = load_policy()
    env, metadata, motion_path = make_environment(apple_rolling_resistance=APPLE_ROLLING_RESISTANCE_M)
    episodes, attempts = [], []
    with torch.inference_mode():
        initial_learner = warmup(env, policy, metadata, motion_path)
        for case in cases:
            modes = ('learned',) if args.skip_first_close else ('control', 'learned')
            for mode in modes:
                episode, rows = run_episode(
                    env, policy, metadata, motion_path, case, mode, args.max_attempts,
                    deepcopy(initial_learner) if mode == 'learned' else None,
                    video if (case, mode) == (args.video_case, args.video_mode) else None,
                    skip_first_close=args.skip_first_close,
                )
                episodes.append(episode)
                attempts.extend(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(summary_path, episodes)
    save_csv(attempts_path, attempts)


if __name__ == '__main__':
    main()
