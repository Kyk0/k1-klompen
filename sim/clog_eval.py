"""Run an mjlab policy evaluation script (mjlab_eval.py, not included here) with the K1 wearing the klompen.

Swaps booster_mjlab's K1 MJCF for mjlab/k1_clogs.xml and raises the motion by the clog sole
(--dz, default 0.0217 m = clog pad bottom -0.0599 m vs bare foot -0.0382 m in the foot frame),
so the reference starts with the clogs on the floor instead of 2 cm inside it.
Needs mjlab and booster_mjlab installed, and MJLAB_EVAL_TOOLS set to the folder that holds mjlab_eval.py.
  MJLAB_EVAL_TOOLS=/path/to/tools python clog_eval.py [--barefoot] [--dz 0.0217] --checkpoint ... --motion ... --out ...
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if os.environ.get("MJLAB_EVAL_TOOLS"):
    sys.path.insert(0, os.environ["MJLAB_EVAL_TOOLS"])

argv = sys.argv[1:]
barefoot = "--barefoot" in argv
argv = [a for a in argv if a != "--barefoot"]
dz = 0.0217
if "--dz" in argv:
    i = argv.index("--dz")
    dz = float(argv[i + 1])
    del argv[i:i + 2]

import mjlab.tasks  # noqa: F401  (registry first, then patch the robot XML)

if not barefoot:
    import booster_mjlab.robots.booster_k1.k1_constants as k1c
    k1c.K1_XML = HERE / "mjlab/k1_clogs.xml"
    i = argv.index("--motion") + 1
    src = Path(argv[i])
    d = dict(np.load(src))
    d["body_pos_w"] = d["body_pos_w"].copy()
    d["body_pos_w"][..., 2] += dz
    dst = HERE / "motions" / f"{src.stem}_clogs_dz{dz * 1000:.0f}mm.npz"
    dst.parent.mkdir(exist_ok=True)
    np.savez(dst, **d)
    argv[i] = str(dst)
    print(f"clogs on: {k1c.K1_XML}, motion {dst}")

import mjlab_eval  # noqa: E402

sys.argv = ["mjlab_eval.py"] + argv
mjlab_eval.main()
