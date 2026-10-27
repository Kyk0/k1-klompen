"""Photo-derived wooden clogs: measured K1 foot/ankle, removable mounting base."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
import trimesh
import manifold3d as md
import cadquery as cq
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull,HalfspaceIntersection

HERE=Path(__file__).resolve().parent;REF=HERE/'reference'
from render_tools import render
OUT=HERE/'models';OUT.mkdir(exist_ok=True)
SUPPORT=17.;CLEARANCE=1.;ASSEMBLY_TRAVEL=120.
PITCH_SAMPLES=[-.87,-.65,-.435,-.2,0,.17,.345]
ROLL_SAMPLES=[-.345,-.17,0,.17,.345]
def U(*ms):return trimesh.boolean.union(ms,engine='manifold')
def D(a,*ms):return trimesh.boolean.difference([a,*ms],engine='manifold')
def I(a,b):return trimesh.boolean.intersection([a,b],engine='manifold')
def clean(m,tol=.00001):
    q=md.Mesh(np.float32(m.vertices),np.uint32(m.faces));q.merge()
    s=md.Manifold(q);assert s.status()==md.Error.NoError
    q=s.simplify(tol).to_mesh()
    return trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts)
def box(lo,hi):
    lo=np.array(lo);hi=np.array(hi)
    m=trimesh.creation.box(hi-lo);m.apply_translation((hi+lo)/2);return m
def cyl(r,z0,z1,x,y):
    m=trimesh.creation.cylinder(r,z1-z0,sections=64)
    m.apply_translation([x,y,(z0+z1)/2]);return m
def cap_trim(x,z):
    curve=PchipInterpolator(x,z)
    samples=np.linspace(x[0],x[-1],220)
    profile=[(float(q),float(curve(q))) for q in samples]
    profile += [(x[-1],210),(x[0],210)]
    solid=cq.Workplane('XZ').polyline(profile).close().extrude(200,both=True).val()
    v,f=solid.tessellate(.04,.12)
    m=trimesh.Trimesh([p.toTuple() for p in v],f)
    m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.fix_normals()
    assert m.is_volume
    return m
def offset_hull(m,d):
    h=ConvexHull(m.vertices);planes=h.equations.copy();planes[:,3]-=d
    v=HalfspaceIntersection(planes,m.vertices.mean(0)).intersections
    solid=md.Manifold.hull_points(np.unique(np.round(v,5),axis=0));q=solid.to_mesh()
    out=trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts)
    assert out.is_volume
    return out,planes
def down_sweep(m,travel):
    points=np.unique(np.round(np.r_[m.vertices,m.vertices-[0,0,travel]],4),axis=0)
    solid=md.Manifold.hull_points(points)
    q=solid.to_mesh()
    out=trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts)
    assert out.is_volume
    return out
def overlap(a,b):return abs(I(a,b).volume)
def rotate_at(m,angle,axis,pivot):
    out=m.copy();out.apply_transform(trimesh.transformations.rotation_matrix(angle,axis,pivot));return out

# Registration is recovered from the prior CAD fit, then corrected so the actual
# mesh's lowest point rests on the new support plane (no floating-foot offset).
reg=np.load(REF/'foot_in_shoe.npz')
swap=np.array([[1,0,0],[0,0,-1],[0,1,0]])
A=np.eye(4);A[:3,:3]=swap@reg['R'];A[:3,3]=swap@reg['t']
T=trimesh.transformations.rotation_matrix(-math.asin(.27146016502209125),[0,1,0])
T[2,3]=16-43.610778808605374;A=T@A
foot=trimesh.load(REF/'Left_Foot.STL',force='mesh');foot.apply_scale(1000);foot.apply_transform(A)
A[2,3]+=SUPPORT-foot.bounds[0,2]
foot=trimesh.load(REF/'Left_Foot.STL',force='mesh');foot.apply_scale(1000);foot.apply_transform(A)
foot_hull=foot.convex_hull
right_foot=trimesh.load(REF/'Right_Foot.STL',force='mesh');right_foot.apply_scale(1000);right_foot.apply_transform(A)
right_foot_normalized=right_foot.copy();right_foot_normalized.vertices[:,1]*=-1
pair_foot_hull=trimesh.convex.convex_hull(np.r_[foot.vertices,right_foot_normalized.vertices])
cross=trimesh.load(REF/'Left_Ankle_Cross.STL',force='mesh');cross.apply_scale(1000);cross.apply_transform(A)
right_cross=trimesh.load(REF/'Right_Ankle_Cross.STL',force='mesh');right_cross.apply_scale(1000);right_cross.apply_transform(A)
right_cross.vertices[:,1]*=-1
pair_cross_hull=trimesh.convex.convex_hull(np.r_[cross.vertices,right_cross.vertices])
shank=trimesh.load(REF/'Left_Shank.STL',force='mesh');shank.apply_scale(1000)
shank.apply_translation(-np.array([.00019706,.0002,-.24519])*1000)
shank.apply_transform(A)
right_shank=trimesh.load(REF/'Right_Shank.STL',force='mesh');right_shank.apply_scale(1000)
right_shank.apply_translation(-np.array([.00019706,-.0002,-.24519])*1000)
right_shank.apply_transform(A);right_shank.vertices[:,1]*=-1
pair_shank_hull=trimesh.convex.convex_hull(np.r_[shank.vertices,right_shank.vertices])
ankle_origin=trimesh.transform_points([[0,0,0]],A)[0]
print('Measured foot mm',foot.extents,'bounds',foot.bounds,flush=True)

# Extract the actual mesh bores at the 5.5 mm circular section. Only the three
# rear mounting bores corresponding to the established screw pattern are used.
section=foot.section(plane_origin=[0,0,A[2,3]-15.0694891],plane_normal=[0,0,1])
mounts=[]
for loop in section.discrete:
    lo=loop[:,:2].min(0);hi=loop[:,:2].max(0);size=hi-lo;center=(lo+hi)/2
    if np.max(abs(size-5.5))<.025 and -25<center[0]<-5 and abs(center[1])<13:
        if not any(np.linalg.norm(center-np.array(q))<.05 for q in mounts):mounts.append(center.tolist())
mounts.sort(key=lambda q:(q[0],q[1]));assert len(mounts)==3,mounts
print('Measured mounting centers',mounts,flush=True)
right_section=right_foot.section(plane_origin=[0,0,A[2,3]-15.0694891],plane_normal=[0,0,1])
right_mounts=[]
for loop in right_section.discrete:
    lo=loop[:,:2].min(0);hi=loop[:,:2].max(0);c=(lo+hi)/2
    if np.max(abs((hi-lo)-5.5))<.025 and -25<c[0]<-5 and abs(c[1])<13:
        if not any(np.linalg.norm(c-np.array(q))<.05 for q in right_mounts):right_mounts.append(c.tolist())
assert len(right_mounts)==3
for x,y in mounts:
    assert min(np.linalg.norm(np.array([x,-y])-q) for q in right_mounts)<.02

cavity,cavity_planes=offset_hull(pair_foot_hull,CLEARANCE)
clearance_min=float(np.min(-(foot.vertices@cavity_planes[:,:3].T+cavity_planes[:,3])))
assert clearance_min>.999
right_clearance_min=float(np.min(-(right_foot_normalized.vertices@cavity_planes[:,:3].T+cavity_planes[:,3])))
assert right_clearance_min>.999
foot_sweep=down_sweep(cavity,ASSEMBLY_TRAVEL)
# Cut a continuous neutral-leg insertion envelope, not merely a final-position
# hole. This permits the upper to descend while the base is already on the foot.
leg_clip=I(pair_shank_hull,box([-200,-200,17],[200,200,250]))
leg_clear,_=offset_hull(leg_clip,1.)
leg_sweep=down_sweep(leg_clear,ASSEMBLY_TRAVEL)
cross_clear,_=offset_hull(pair_cross_hull,1.)
cross_sweep=down_sweep(cross_clear,ASSEMBLY_TRAVEL)

