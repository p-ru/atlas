"""Inspect the two-attempt floor pick in the native MuJoCo viewer."""

import argparse
import sys
import threading
import time
from pathlib import Path

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
from loguru import logger

from floor_pick import make_environment
from floor_pick_retry import APPLE_ROLLING_RESISTANCE_M, STATE_SPEC, run_episode, warmup
from scalebfm_playback import load_policy, output

DEFAULT_REPLAY = output / 'scalebfm_floor_pick_retry_replay.npz'


def capture(env, metadata, motion_path, path):
    sim = env.simulator
    state_size = mj.mj_stateSize(sim.mujoco_model, STATE_SPEC)
    frames = []

    def capture_frame():
        state = np.empty(state_size)
        mj.mj_getState(sim.mujoco_model, sim.mujoco_data, state, STATE_SPEC)
        frames.append(state)

    policy = load_policy()
    with torch.inference_mode():
        learner = warmup(env, policy, metadata, motion_path)
        result, attempts = run_episode(
            env, policy, metadata, motion_path, 'right', 'learned', 3, learner, None,
            skip_first_close=True, frame_hook=capture_frame,
        )
    if len(attempts) != 2 or attempts[0]['lifted'] or not result['episode_lift']:
        raise RuntimeError('The recorded episode did not reproduce the miss and retry')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, frames=np.stack(frames), frame_period_s=2 * env.dt,
        retry_time_s=attempts[1]['start_time_s'],
    )


def show(sim, path):
    with np.load(path) as archive:
        frames = archive['frames']
        period = float(archive['frame_period_s'])
        retry_time = float(archive['retry_time_s'])
    model, data = sim.mujoco_model, sim.mujoco_data
    if frames.ndim != 2 or frames.shape[1] != mj.mj_stateSize(model, STATE_SPEC):
        raise ValueError('Replay does not match this MuJoCo model')

    frame = 0
    playing = False
    retry_frame = round(retry_time / period)
    lock = threading.Lock()

    def key_callback(key):
        nonlocal frame, playing
        with lock:
            if key == glfw.KEY_SPACE:
                if not playing and frame == len(frames) - 1:
                    frame = 0
                playing = not playing
            elif key == glfw.KEY_R:
                frame, playing = 0, False
            elif key == glfw.KEY_T:
                frame, playing = retry_frame, False
            elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
                step = -1 if key == glfw.KEY_LEFT_BRACKET else 1
                frame = max(0, min(len(frames) - 1, frame + step))
                playing = False

    mj.mj_setState(model, data, frames[0], STATE_SPEC)
    mj.mj_forward(model, data)
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, -.08, .55]
            viewer.cam.distance = 1.9
            viewer.cam.azimuth = 122
            viewer.cam.elevation = -15
        print('Native viewer open. Space: play/pause | T: retry | R: restart | [ ]: step', flush=True)
        displayed = -1
        playback_status = None
        next_frame_at = time.monotonic() + period
        while viewer.is_running():
            now = time.monotonic()
            with lock:
                current, active = frame, playing
                if active and now >= next_frame_at:
                    frame = min(frame + 1, len(frames) - 1)
                    playing = frame < len(frames) - 1
                    next_frame_at = now + period
                elif not active:
                    next_frame_at = now + period
            if current != displayed:
                with viewer.lock():
                    mj.mj_setState(model, data, frames[current], STATE_SPEC)
                    mj.mj_forward(model, data)
                displayed = current
            status = (current, active)
            if status != playback_status:
                attempt = 'Attempt 1' if current < retry_frame else 'Attempt 2'
                viewer.set_texts([
                    (None, None, 'Floor pick retry', f'{attempt}   {current * period:.2f} s'),
                    (None, None, 'Playing' if active else 'Paused', 'Space: play/pause   T: retry   R: restart   [ ]: step'),
                ])
                playback_status = status
            viewer.sync()
            time.sleep(.008)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=DEFAULT_REPLAY)
    parser.add_argument('--record', action='store_true', help='Recreate a cached trajectory')
    args = parser.parse_args()
    path = args.replay.resolve()
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    env, metadata, motion_path = make_environment(apple_rolling_resistance=APPLE_ROLLING_RESISTANCE_M)
    if args.record or not path.is_file():
        capture(env, metadata, motion_path, path)
    show(env.simulator, path)


if __name__ == '__main__':
    main()
