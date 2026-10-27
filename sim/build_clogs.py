#!/usr/bin/env python3
"""Mount the Dutch klompen (v10 "Smooth Toe" by default) on the K1 feet; emit URDF + MJCF.

Inputs:
  ../design/Print_Shoes/K1_Left_Klomp_One_Piece.stl   printing STL, mm, Z up, toes +X, sole at Z=0
                                            (--shoe PATH to use another left-shoe STL, e.g. the v11 one)
  ../design/reference/foot_in_shoe.npz + Left_Foot.STL   registration used by the shoe's fit_helpers.py
  K1_22dof.urdf / K1_22dof.xml / meshes/    official Booster K1 model (this folder)
  BOOSTER_K1_XML (env var, optional)        path to booster_mjlab's booster_k1/xmls/k1.xml (mjlab variant)

Outputs:
  meshes/{Left,Right}_Klomp.STL             visual mesh, metres, ankle-roll (foot) link frame
  meshes/{Left,Right}_Klomp_Collision.STL   convex hull, bottom lowered to the rubber pads (-4.7 mm)
  K1_Left_Klomp.urdf                        one shoe alone
  K1_22dof_clogs.urdf / K1_22dof_clogs.xml  official K1 with the clogs (MJCF has a "stand" key)
  mjlab/k1_clogs.xml                        booster_mjlab K1 with the clogs (shoe collision is named
                                            <side>_foot_collision so the task's foot rules still apply);
                                            only written when BOOSTER_K1_XML is set. Its meshes are read
                                            from ../meshes/ (any K1 mesh missing there is copied in).

Usage: [BOOSTER_K1_XML=/path/to/k1.xml] python build_clogs.py [mass_kg_per_shoe=0.35] [--shoe LEFT_SHOE.stl]
"""
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent / "design"
SHOE_STL = DESIGN / "Print_Shoes/K1_Left_Klomp_One_Piece.stl"
BOOSTER_K1_XML = Path(os.environ["BOOSTER_K1_XML"]) if os.environ.get("BOOSTER_K1_XML") else None
SUPPORT_MM = 17.0     # fit_helpers.SUPPORT: foot rests on Z=17 in the shoe
PAD_Z_MM = -4.7       # rubber pads project 4.7 mm below the sole (README / validation.json)
HULL_PTS = 48         # farthest-point samples kept in the collision hull (+ ~24 sole points)
RGBA = "0.80 0.62 0.33 1"  # wood
STAND = {  # booster_mjlab HOME_KEYFRAME (official joint names); flat sole, so no pitch change
    "left_shoulder_roll_joint": -1.4, "left_elbow_yaw_joint": -0.4,
    "right_shoulder_roll_joint": 1.4, "right_elbow_yaw_joint": 0.4,
    "left_hip_pitch_joint": -0.4, "left_knee_pitch_joint": 0.8, "left_ankle_pitch_joint": -0.4,
    "right_hip_pitch_joint": -0.4, "right_knee_pitch_joint": 0.8, "right_ankle_pitch_joint": -0.4,
}


def foot_to_shoe():
    """4x4 foot(mm) -> shoe(mm), exactly as in the shoe's fit_helpers.py."""
    reg = np.load(DESIGN / "reference/foot_in_shoe.npz")
    swap = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    A = np.eye(4)
    A[:3, :3] = swap @ reg["R"]
    A[:3, 3] = swap @ reg["t"]
    T = trimesh.transformations.rotation_matrix(-math.asin(.27146016502209125), [0, 1, 0])
    T[2, 3] = 16 - 43.610778808605374
    A = T @ A
    foot = trimesh.load(DESIGN / "reference/Left_Foot.STL", force="mesh")
    foot.apply_scale(1000)
    foot.apply_transform(A)
    A[2, 3] += SUPPORT_MM - foot.bounds[0, 2]
    return A


def mirror_y(m):
    m = m.copy()
    m.vertices[:, 1] *= -1
    m.invert()
    return m


def build_meshes(mass):
    to_foot = np.linalg.inv(foot_to_shoe())
    to_foot[:3, :] /= 1000.0  # mm -> m (rotation part scaled too, fixed below)
    to_foot[:3, :3] *= 1000.0
    shoe = trimesh.load(SHOE_STL, force="mesh")
    sole = shoe.vertices[shoe.vertices[:, 2] < 0.5].copy()
    sole[:, 2] = PAD_Z_MM
    hull = trimesh.convex.convex_hull(np.r_[shoe.vertices, sole])
    # mujoco_warp needs a small hull: keep HULL_PTS farthest-point samples of the full hull,
    # always including the sole corners (lowest ring) so the contact patch keeps its footprint.
    pts = hull.vertices
    low = pts[pts[:, 2] < PAD_Z_MM + 0.1]
    keep = [int(np.argmin(pts[:, 0]))]
    dist = np.linalg.norm(pts - pts[keep[0]], axis=1)
    for _ in range(HULL_PTS - 1):
        keep.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(pts - pts[keep[-1]], axis=1))
    hull = trimesh.convex.convex_hull(np.r_[pts[keep], low[::max(1, len(low) // 24)]])
    left, left_col = shoe.copy(), hull
    for m in (left, left_col):
        m.vertices = m.vertices / 1000.0
        m.apply_transform(np.r_[np.c_[to_foot[:3, :3], to_foot[:3, 3]], [[0, 0, 0, 1]]])
    out = HERE / "meshes"
    left.export(out / "Left_Klomp.STL")
    mirror_y(left).export(out / "Right_Klomp.STL")
    left_col.export(out / "Left_Klomp_Collision.STL")
    mirror_y(left_col).export(out / "Right_Klomp_Collision.STL")
    m = left.copy()
    m.density = mass / m.volume
    com, I = m.center_mass, m.moment_inertia  # inertia about the COM
    print(f"shoe in foot frame bounds (m):\n{left.bounds.round(4)}\ncollision bottom z {left_col.bounds[0, 2]:.4f} m")
    print(f"mass {mass} kg, COM {com.round(4)}")
    return com, I


def inertial(side, com, I, mass, fmt):
    s = -1 if side == "right" else 1
    c = com * [1, s, 1]
    ixx, iyy, izz, ixy, ixz, iyz = I[0, 0], I[1, 1], I[2, 2], s * I[0, 1], I[0, 2], s * I[1, 2]
    if fmt == "urdf":
        return (f'<inertial>\n      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0" />\n'
                f'      <mass value="{mass:.4f}" />\n'
                f'      <inertia ixx="{ixx:.4e}" ixy="{ixy:.4e}" ixz="{ixz:.4e}" iyy="{iyy:.4e}" iyz="{iyz:.4e}" izz="{izz:.4e}" />\n'
                f'    </inertial>')
    return (f'<inertial pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" mass="{mass:.4f}" '
            f'fullinertia="{ixx:.4e} {iyy:.4e} {izz:.4e} {ixy:.4e} {ixz:.4e} {iyz:.4e}"/>')


def urdf_link(side, com, I, mass, parent):
    S = side.capitalize()
    return f"""  <link name="{side}_klomp_link">
    {inertial(side, com, I, mass, "urdf")}
    <visual>
      <geometry><mesh filename="meshes/{S}_Klomp.STL" /></geometry>
      <material name="klomp_wood"><color rgba="{RGBA}" /></material>
    </visual>
    <collision>
      <geometry><mesh filename="meshes/{S}_Klomp_Collision.STL" /></geometry>
    </collision>
  </link>
  <joint name="{side}_klomp_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="{parent}" />
    <child link="{side}_klomp_link" />
  </joint>
"""


def build_urdf(com, I, mass):
    (HERE / "K1_Left_Klomp.urdf").write_text(
        '<?xml version="1.0"?>\n<!-- K1 klomp (left). Frame = K1 left_ankle_roll_link, metres. -->\n'
        '<robot name="K1_Left_Klomp">\n'
        + urdf_link("left", com, I, mass, "PARENT").split("  <joint")[0] + "</robot>\n")
    src = (HERE / "K1_22dof.urdf").read_text()
    for side in ("left", "right"):
        # the bare foot sits inside the clog: drop its collision, keep its visual
        src, n = re.subn(rf"(<link\s+name=\"{side}_ankle_roll_link\">.*?)<collision>.*?</collision>",
                         r"\1", src, count=1, flags=re.S)
        end = re.search(rf"<joint\s+name=\"{side}_ankle_roll_joint\".*?</joint>\s*\n", src, re.S).end()
        src = src[:end] + urdf_link(side, com, I, mass, f"{side}_ankle_roll_link") + src[end:]
    (HERE / "K1_22dof_clogs.urdf").write_text(src)


def shoe_body(side, com, I, mass, col_name, indent, vis_group):
    S = side.capitalize()
    p = " " * indent
    return (f'\n{p}<body name="{side}_klomp_link">\n'
            f'{p}  {inertial(side, com, I, mass, "mjcf")}\n'
            f'{p}  <geom name="{side}_klomp_visual" type="mesh" contype="0" conaffinity="0" group="{vis_group}" '
            f'density="0" rgba="{RGBA}" mesh="{S}_Klomp"/>\n'
            f'{p}  <geom name="{col_name}" type="mesh" mesh="{S}_Klomp_Collision" density="0" '
            f'rgba="{RGBA}" group="3"/>\n'
            f'{p}</body>')


def mesh_assets(prefix=""):
    return "".join(f'\n    <mesh name="{S}_Klomp{c}" file="{prefix}{S}_Klomp{c}.STL"/>'
                   for S in ("Left", "Right") for c in ("", "_Collision"))


def stand_key(src, geoms):
    import mujoco
    model = mujoco.MjModel.from_xml_string(src.replace('meshdir="meshes/"', f'meshdir="{HERE}/meshes/"'))
    qpos = model.qpos0.copy()
    for j, v in STAND.items():
        qpos[model.jnt_qposadr[model.joint(j).id]] = v
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_forward(model, d)
    zmin = min(d.geom_xpos[g][2] + (d.geom_xmat[g].reshape(3, 3) @ model.mesh_vert[
        model.mesh_vertadr[model.geom_dataid[g]]:][:model.mesh_vertnum[model.geom_dataid[g]]].T)[2].min()
        for g in (model.geom(n).id for n in geoms))
    qpos[2] -= zmin - 0.001
    print(f"stand base height {qpos[2]:.4f} m, total mass {model.body_subtreemass[1]:.3f} kg")
    return qpos


def build_mjcf(com, I, mass):
    src = (HERE / "K1_22dof.xml").read_text().replace('<mujoco model="K1">', '<mujoco model="K1_clogs">')
    src = src.replace('<mesh name="Right_Foot" file="Right_Foot.STL"/>',
                      '<mesh name="Right_Foot" file="Right_Foot.STL"/>' + mesh_assets())
    for side, S in (("left", "Left"), ("right", "Right")):
        pat = rf'(mesh="{S}_Foot"/>.*?)\n\s*<geom type="box" pos="0.026 0 -0.02" size="0.09 0.035 0.018"[^>]*/>'
        src, n = re.subn(pat, lambda m: m.group(1) + shoe_body(side, com, I, mass, f"{side}_klomp", 18, 1),
                         src, count=1, flags=re.S)
        assert n == 1, side
    qpos = stand_key(src, ("left_klomp", "right_klomp"))
    src = src.replace("\n</mujoco>", f'\n  <keyframe>\n    <key name="stand" qpos="{" ".join(f"{x:.5g}" for x in qpos)}"/>\n'
                                     f'  </keyframe>\n</mujoco>')
    (HERE / "K1_22dof_clogs.xml").write_text(src)


def build_mjlab(com, I, mass):
    src = BOOSTER_K1_XML.read_text().replace('<mujoco model="K1"', '<mujoco model="K1_clogs"', 1)
    # Relative paths only: the K1 meshes booster_mjlab uses are the ones in ../meshes/ (copy any missing).
    for f in set(re.findall(r'file="([^"/]+\.STL)"', src)):
        if not (HERE / "meshes" / f).exists():
            shutil.copy(BOOSTER_K1_XML.parent / "meshes" / f, HERE / "meshes" / f)
    src = src.replace('meshdir="meshes/"', 'meshdir="../meshes/"')
    src = re.sub(r'(<mesh name="Right_Foot" file="Right_Foot.STL" ?/>)',
                 lambda m: m.group(1) + mesh_assets(), src, count=1)
    import mujoco
    for side in ("left", "right"):
        S, s = side.capitalize(), (-1 if side == "right" else 1)
        # No extra body here: mjlab motion files index the robot's bodies, so the clog geoms go on
        # the foot link and the clog mass is merged into the foot's inertial.
        pat = rf'<inertial pos="([^"]+)" quat="([^"]+)" mass="([^"]+)" diaginertia="([^"]+)" />(\s*<joint name="{S}_Ankle_Roll")'
        mm = re.search(pat, src)
        fp, fq, fm, fd = (np.array(mm.group(i).split(), float) for i in range(1, 5))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, fq)
        R = R.reshape(3, 3)
        c = com * [1, s, 1]
        Is = I * np.outer([1, s, 1], [1, s, 1])
        M = fm[0] + mass
        C = (fm[0] * fp + mass * c) / M
        par = lambda m_, p: m_ * ((p - C) @ (p - C) * np.eye(3) - np.outer(p - C, p - C))
        It = R @ np.diag(fd) @ R.T + par(fm[0], fp) + Is + par(mass, c)
        new = (f'<inertial pos="{C[0]:.6f} {C[1]:.6f} {C[2]:.6f}" mass="{M:.4f}" fullinertia="'
               f'{It[0, 0]:.4e} {It[1, 1]:.4e} {It[2, 2]:.4e} {It[0, 1]:.4e} {It[0, 2]:.4e} {It[1, 2]:.4e}" />'
               f'<!-- foot 0.494 kg + klomp {mass} kg -->')
        src = src[:mm.start()] + new + mm.group(5) + src[mm.end():]
        # bare-foot collision -> the clog (same name, so friction/condim/contact sensors carry over)
        pat = rf'<geom name="{side}_foot_collision" type="mesh"[^>]*/>'
        geoms = (f'<geom name="{side}_klomp_visual" type="mesh" contype="0" conaffinity="0" group="2" '
                 f'density="0" rgba="{RGBA}" mesh="{S}_Klomp" />\n                  '
                 f'<geom name="{side}_foot_collision" type="mesh" mesh="{S}_Klomp_Collision" '
                 f'rgba="{RGBA}" group="3" />')
        src, n = re.subn(pat, lambda _: geoms, src, count=1)
        assert n == 1, side
    (HERE / "mjlab").mkdir(exist_ok=True)
    (HERE / "mjlab/k1_clogs.xml").write_text(src)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(HERE / "mjlab/k1_clogs.xml"))
    print(f"mjlab model ok, total mass {m.body_subtreemass[1]:.3f} kg")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--shoe" in args:
        i = args.index("--shoe")
        SHOE_STL = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    mass = float(args[0]) if args else 0.35
    com, I = build_meshes(mass)
    build_urdf(com, I, mass)
    build_mjcf(com, I, mass)
    if BOOSTER_K1_XML is not None:
        build_mjlab(com, I, mass)
    else:
        print("BOOSTER_K1_XML not set: skipping mjlab/k1_clogs.xml")
    print("done")
