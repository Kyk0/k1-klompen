# K1 with the klompen (simulation)

The Booster K1 wearing the v10 "Smooth Toe" clogs. Each clog is 0.35 kg in the model; the robot stands 2.2 cm taller; the K1 with clogs weighs 20.37 kg in `K1_22dof_clogs.xml`.

- `K1_22dof_clogs.xml`: MuJoCo model with a "stand" keyframe. `K1_22dof_clogs.urdf`: the same as URDF. `K1_Left_Klomp.urdf`: one shoe alone.
- `mjlab/k1_clogs.xml`: booster_mjlab K1 variant for mjlab. Clog collisions are named `<side>_foot_collision`, and the clog mass is merged into the foot link. It uses the meshes in `meshes/`.
- `K1_22dof.xml`, `K1_22dof.urdf`: the K1 without clogs (Booster model).
- `meshes/`: K1 meshes (Booster) and the clog meshes (`*_Klomp.STL` visual, `*_Klomp_Collision.STL` convex hull, in the ankle-roll link frame, metres).
- `renders/`: stand-pose stills from `render_stills.py`.
- `videos/`: a dance policy trained barefoot, run barefoot and in the clogs. It does not fall in the clogs; body tracking error is 4.94 cm in the clogs vs 4.86 cm barefoot.

Scripts:

- `view.py`: open the model in the MuJoCo viewer (`mjpython view.py` on macOS).
- `build_clogs.py`: rebuild the clog meshes, URDF and MJCF from the print STL in `../design/Print_Shoes/` (`--shoe PATH` for another STL). Set `BOOSTER_K1_XML` to a booster_mjlab `k1.xml` to also write `mjlab/k1_clogs.xml`.
- `render_stills.py`, `render_rollout.py`: still images and rollout videos.
- `clog_eval.py`: run an mjlab policy evaluation with the clogs on (needs mjlab, booster_mjlab and an evaluation script that is not included; see the file header).
