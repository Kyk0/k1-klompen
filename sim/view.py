"""Open the K1 + v10 klompen MJCF in the MuJoCo viewer, standing on a PD hold of the "stand" keyframe.
   mjpython view.py [K1_22dof_clogs.xml]      (macOS needs mjpython for the passive viewer)"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path(sys.argv[1] if len(sys.argv) > 1 else "K1_22dof_clogs.xml")
d = mujoco.MjData(m)
key = m.key("stand").id
mujoco.mj_resetDataKeyframe(m, d, key)
q_adr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
v_adr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
q_ref = m.key_qpos[key][q_adr]
KP, KD = 60.0, 2.0
with mujoco.viewer.launch_passive(m, d) as v:
    v.opt.geomgroup[3] = 0  # hide collision hulls (toggle with key 3)
    while v.is_running():
        t0 = time.time()
        for _ in range(int(0.01 / m.opt.timestep)):
            d.ctrl[:] = KP * (q_ref - d.qpos[q_adr]) - KD * d.qvel[v_adr]
            mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0, 0.01 - (time.time() - t0)))
