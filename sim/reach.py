"""Test online Cartesian residual correction using a fixed-base G1 in MuJoCo."""

import argparse
import csv
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np

ARM = tuple(f"right_{joint}_joint" for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
))
BIAS_M = np.array([-0.079, -0.041, 0.0])  # G1 frame: +x forward, +y left
WARMUP = 5
REACH_STEPS = 40
HOLD_STEPS = 25
RESIDUAL_LIMIT_M = 0.12  # per axis
RESIDUAL_RATE_M = 0.006  # per axis per 20 ms
HIT_RADIUS_M = 0.025
HORIZON = 12
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "gensong_gmr/TWIST2-master/assets/g1/g1_sim2sim_29dof.xml"
)


class G1Arm:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str(path.parent / compiler.get("meshdir")))
        root.find("option").set("integrator", "implicitfast")
        for body in root.iter("body"):
            for joint in list(body):
                if joint.tag in ("joint", "freejoint") and joint.get("name") not in ARM:
                    body.remove(joint)
        actuators = root.find("actuator")
        for actuator in list(actuators):
            if actuator.get("joint") not in ARM:
                actuators.remove(actuator)
        root.remove(root.find("keyframe"))
        hand = root.find(".//body[@name='right_rubber_hand']")
        ET.SubElement(hand, "site", name="reach_contact", pos="0.065 0 0", size="0.008")
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mj.MjData(self.model)
        self.ik = mj.MjData(self.model)
        self.site = self.model.site("reach_contact").id
        self.jacobian = np.empty((3, self.model.nv))
        self.damping = np.array([3.0, 2.5, 1.7, 1.7, 0.12, 0.12, 0.08])
        self.home = np.array([0.05, -0.25, 0, 0.45, 0, 0, 0])
        self.reset()
        self.start = self.position()
        self.ik.qpos[:] = [-0.18, -0.38, 0.20, 0.75, 0, 0.40, 0]
        mj.mj_fwdPosition(self.model, self.ik)
        self.target = self.ik.site_xpos[self.site].copy()

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home
        mj.mj_forward(self.model, self.data)
        self.ik.qpos[:] = self.home

    def position(self):
        return self.data.site_xpos[self.site].copy()

    def step(self, command):
        for _ in range(12):
            mj.mj_fwdPosition(self.model, self.ik)
            error = command - self.ik.site_xpos[self.site]
            if np.linalg.norm(error) < 0.0003:
                break
            mj.mj_jacSite(self.model, self.ik, self.jacobian, None, self.site)
            change = self.jacobian.T @ np.linalg.solve(
                self.jacobian @ self.jacobian.T + 0.001 * np.eye(3), error
            )
            self.ik.qpos[:] = np.clip(
                self.ik.qpos + np.clip(change, -0.20, 0.20),
                self.model.jnt_range[:, 0], self.model.jnt_range[:, 1],
            )
        for _ in range(round(0.02 / self.model.opt.timestep)):
            # Position tracking with gravity and velocity-bias feedforward.
            self.data.ctrl[:] = (
                75 * (self.ik.qpos - self.data.qpos)
                - self.damping * self.data.qvel + self.data.qfrc_bias
            )
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)


class ReachDynamics:
    """Fit c[t+1] from current contact and pre-bias command after each attempt."""

    def __init__(self, origin):
        self.origin = origin
        self.gram = 1e-5 * np.eye(7)
        self.cross = np.zeros((7, 3))

    def observe(self, transitions):
        features = np.array([
            np.r_[contact - self.origin, command - self.origin, 1]
            for contact, command, _ in transitions
        ])
        next_contact = np.array([after - self.origin for _, _, after in transitions])
        self.gram += features.T @ features
        self.cross += features.T @ next_contact

    def fit(self):
        return np.linalg.solve(self.gram, self.cross)

    def residual(self, weights, contact, nominal, previous):
        # Predict over a short horizon with one candidate offset held constant.
        # At each tick only the first command is applied before observing again.
        state = contact - self.origin
        command = nominal - self.origin
        offset_gain = np.zeros((3, 3))
        for _ in range(HORIZON):
            state = state @ weights[:3] + command @ weights[3:6] + weights[6]
            offset_gain = offset_gain @ weights[:3] + weights[3:6]
        penalty = 0.03
        offset = np.linalg.solve(
            offset_gain @ offset_gain.T + penalty * np.eye(3),
            offset_gain @ (command - state) + penalty * previous,
        )
        return np.clip(
            offset,
            np.maximum(-RESIDUAL_LIMIT_M, previous - RESIDUAL_RATE_M),
            np.minimum(RESIDUAL_LIMIT_M, previous + RESIDUAL_RATE_M),
        )


def session(arm, seed, attempts, bias_std_m):
    rng = np.random.default_rng(seed)
    dynamics = ReachDynamics(arm.start)
    for attempt in range(1, attempts + 1):
        arm.reset()
        mode = "warmup" if attempt <= WARMUP else rng.choice(
            ["control", "model", "explore"]
        )
        target = arm.target + rng.uniform(-0.012, 0.012, size=3)
        bias = BIAS_M + bias_std_m * rng.standard_normal(3)
        exploration = rng.uniform(-0.105, 0.105, size=3)
        weights = dynamics.fit() if mode == "model" else None
        residual = np.zeros(3)
        transitions = []
        hold_errors = []
        for tick in range(1, REACH_STEPS + HOLD_STEPS + 1):
            fraction = min(tick / REACH_STEPS, 1)
            smooth = fraction**2 * (3 - 2 * fraction)
            nominal = arm.start + smooth * (target - arm.start)
            contact = arm.position()
            if mode == "model":
                residual = dynamics.residual(weights, contact, nominal, residual)
            elif mode in ("warmup", "explore"):
                residual += np.clip(
                    exploration - residual, -RESIDUAL_RATE_M, RESIDUAL_RATE_M
                )
            command = nominal + residual
            arm.step(command + bias)
            after = arm.position()
            transitions.append((contact, command, after))
            if tick > REACH_STEPS:
                hold_errors.append(np.linalg.norm(after - target))
        dynamics.observe(transitions)
        error = statistics.median(hold_errors)
        yield {
            "seed": seed,
            "attempt": attempt,
            "mode": mode,
            "hold_median_error_cm": 100 * error,
            "within_2_5_cm": int(error <= HIT_RADIUS_M),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--attempts", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--bias-std-m", type=float, default=0,
        help="Optional synthetic per-axis per-attempt bias standard deviation in meters",
    )
    parser.add_argument("--output", type=Path, default=Path("results/fixed_bias.csv"))
    args = parser.parse_args()
    if args.attempts < 1 or not np.isfinite(args.bias_std_m) or args.bias_std_m < 0:
        parser.error("attempts must be positive and bias standard deviation finite and nonnegative")
    if not args.mjcf.is_file():
        parser.error(f"G1 MJCF not found at {args.mjcf}; specify --mjcf with its meshes nearby")

    arm = G1Arm(args.mjcf.resolve())
    rows = [row for seed in args.seeds for row in session(
        arm, seed, args.attempts, args.bias_std_m
    )]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for mode in ("model", "control"):
        scored = [row for row in rows if row["mode"] == mode]
        if scored:
            error = statistics.median(row["hold_median_error_cm"] for row in scored)
            hits = sum(row["within_2_5_cm"] for row in scored)
            print(f"{mode}: median hold error {error:.2f} cm; within 2.5 cm {hits}/{len(scored)}")
    print(f"Saved {len(rows)} attempts to {args.output}")


if __name__ == "__main__":
    main()
