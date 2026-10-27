"""Still renders of the K1 + klompen in the "stand" keyframe (MuJoCo look): full body + feet close-ups.
   python render_stills.py [K1_22dof_clogs.xml] [outdir=renders]"""
import sys
from pathlib import Path

import mujoco
from PIL import Image

try:  # optional scene styling helper (not included); plain MuJoCo scene otherwise
    from mj_scene import mujoco_theme  # noqa: E402
except ImportError:
    def mujoco_theme(xml, w, h):
        m = mujoco.MjModel.from_xml_path(str(xml))
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, m.vis.global_.offwidth), max(h, m.vis.global_.offheight)
        return m

xml = sys.argv[1] if len(sys.argv) > 1 else "K1_22dof_clogs.xml"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "renders")
out.mkdir(exist_ok=True)
W, H = 1600, 1200
m = mujoco_theme(xml, W, H)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 0
views = {  # name: (lookat z, distance, elevation, azimuth)
    "full_front_three_quarter": (0.40, 1.9, -10, 145),
    "full_side": (0.40, 1.9, -5, 90),
    "feet_front_three_quarter": (0.06, 0.65, -18, 145),
    "feet_side": (0.05, 0.55, -6, 90),
    "feet_top": (0.04, 0.70, -70, 180),
    "feet_back": (0.06, 0.65, -15, 330),
}
for name, (z, dist, el, az) in views.items():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [d.qpos[0] + 0.03, d.qpos[1], z]
    cam.distance, cam.elevation, cam.azimuth = dist, el, az
    r.update_scene(d, cam, opt)
    Image.fromarray(r.render()).save(out / f"{name}.png")
    print("wrote", out / f"{name}.png")
