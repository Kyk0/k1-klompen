"""Render an mjlab rollout CSV (xyz, quat xyzw, 22 joints in mjlab order) on the official K1 MJCF.
   python render_rollout.py ROLLOUT.csv OUT.mp4 [--xml K1_22dof_clogs.xml] [--title TEXT]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

try:  # optional scene styling helper (not included); plain MuJoCo scene otherwise
    from mj_scene import mujoco_theme  # noqa: E402
except ImportError:
    def mujoco_theme(xml, w, h):
        m = mujoco.MjModel.from_xml_path(str(xml))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, m.vis.global_.offwidth), max(h, m.vis.global_.offheight)
        return m

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("out")
ap.add_argument("--xml", default="K1_22dof_clogs.xml")
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--size", default="960x720")
a = ap.parse_args()
W, H = map(int, a.size.split("x"))

names = json.loads(Path(a.csv.replace(".rollout.csv", ".json").replace(".ref.csv", ".json")).read_text())["joint_names"]
m = mujoco_theme(a.xml, W, H)
d = mujoco.MjData(m)
def official(n):
    n = n.lower()
    return {"head_yaw": "aahead_yaw_joint", "head_pitch": "aahead_pitch_joint",
            "left_shoulder_pitch": "aaleft_shoulder_pitch_joint", "right_shoulder_pitch": "aaright_shoulder_pitch_joint"}.get(n, n + "_joint")
adr = [m.jnt_qposadr[m.joint(official(n)).id] for n in names]
rows = np.loadtxt(a.csv, delimiter=",")

r = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 0
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 2.2, -12, 150
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(a.fps), "-i", "-", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                      stdin=subprocess.PIPE)
for row in rows:
    d.qpos[:3] = row[:3]
    d.qpos[3:7] = row[[6, 3, 4, 5]]  # xyzw -> wxyz
    d.qpos[adr] = row[7:]
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.45]
    r.update_scene(d, cam, opt)
    ff.stdin.write(r.render().tobytes())
ff.stdin.close()
ff.wait()
print("wrote", a.out, len(rows), "frames")
