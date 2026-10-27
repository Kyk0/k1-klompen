"""One-piece klompen (closed-toe Dutch clogs) with pockets for the unchanged existing heel-shoe pads."""
from pathlib import Path
import math,json,hashlib
import numpy as np
import trimesh
import cadquery as cq
import manifold3d as md
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from render_tools import render

HERE=Path(__file__).resolve().parent
ctx={'__file__':str(HERE/'fit_helpers.py'),'__name__':'fit_measurements'}
exec(compile((HERE/'fit_helpers.py').read_text(),str(HERE/'fit_helpers.py'),'exec'),ctx)
for name in ['U','D','I','box','cyl','clean','offset_hull','down_sweep','overlap','rotate_at','cap_trim']:
    globals()[name]=ctx[name]
PRINT=HERE/'Print_Shoes';PRINT.mkdir(exist_ok=True)
for name in ['pair_foot_hull','pair_cross_hull','pair_shank_hull']:
    q=md.Manifold.hull_points(ctx[name].vertices).to_mesh()
    ctx[name]=trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts)
    assert ctx[name].is_volume

def from_md(s):
    assert s.status()==md.Error.NoError
    q=s.simplify(.0001).to_mesh64();m=trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts)
    m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices();m.fix_normals()
    assert m.is_volume
    return m
def to_md(m):
    q=md.Mesh(np.float32(m.vertices),np.uint32(m.faces));q.merge()
    return md.Manifold(q)
def cqmesh(s):
    v,f=s.tessellate(.035,.12)
    m=trimesh.Trimesh([p.toTuple() for p in v],f)
    m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices();m.fix_normals()
    assert m.is_volume
    return m
def write_mesh(m,path):
    m.export(path);q=trimesh.load(path,force='mesh')
    assert q.is_volume and len(q.split())==1,path.name
    return q
def hull_of(*ms):
    return from_md(md.Manifold.hull_points(np.concatenate([m.vertices for m in ms])))
def union_md(ms):
    s=md.Manifold.batch_boolean([to_md(m) for m in ms],md.OpType.Add)
    assert s.status()==md.Error.NoError
    q=s.simplify(1e-4).to_mesh64();m=trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts,process=False)
    m.fix_normals()
    assert m.is_volume,(m.is_watertight,m.is_winding_consistent,len(m.faces))
    return m
def heightfield_cutter(xg,yg,zfun,ztop=300):
    """Closed solid above z=zfun(x,y) over a rectangular grid."""
    X,Y=np.meshgrid(xg,yg,indexing='ij');Z=zfun(X,Y);nx,ny=X.shape
    bot=np.c_[X.ravel(),Y.ravel(),Z.ravel()];top=bot.copy();top[:,2]=ztop
    verts=np.r_[bot,top];o=nx*ny;faces=[]
    idx=lambda i,j:i*ny+j
    for i in range(nx-1):
        for j in range(ny-1):
            a,b,c,d=idx(i,j),idx(i+1,j),idx(i+1,j+1),idx(i,j+1)
            faces += [(a,c,b),(a,d,c),(a+o,b+o,c+o),(a+o,c+o,d+o)]
    ring=[idx(i,0) for i in range(nx)]+[idx(nx-1,j) for j in range(1,ny)]
    ring+=[idx(i,ny-1) for i in range(nx-2,-1,-1)]+[idx(0,j) for j in range(ny-2,0,-1)]
    for p,q in zip(ring,ring[1:]+ring[:1]):faces += [(p,q,q+o),(p,q+o,p+o)]
    m=trimesh.Trimesh(verts,faces);m.fix_normals();assert m.is_volume
    return m

# The foot top falls from Z 76 at the ankle to Z 46 at the toe. A single convex
# hull bridges that profile and would cut the vamp away, so the foot envelope is
# the union of convex hulls of four x-slabs of both measured feet.
FOOT_CUTS=[-4.,44.,86.]
def foot_slabs(meshes):
    edges=[-1e3]+FOOT_CUTS+[1e3];pieces=[]
    for lo,hi in zip(edges,edges[1:]):
        pts=[]
        for m in meshes:
            q=m.slice_plane([lo,0,0],[1,0,0]).slice_plane([hi,0,0],[-1,0,0]);pts.append(q.vertices)
        pieces.append(from_md(md.Manifold.hull_points(np.concatenate(pts))))
    return pieces
foot_pieces=foot_slabs([ctx['foot'],ctx['right_foot_normalized']])
foot_env=union_md(foot_pieces)
def piece_offset(pieces,d):
    return union_md([offset_hull(p,d)[0] for p in pieces])
cavity_env=piece_offset(foot_pieces,1.)

# Klomp outer envelope around the 185.4 mm foot. The rigid ground lands, arch
# relief and pad locations are unchanged from v4 (width is never narrower over
# the pad lands). The toe is now a closed vamp that tapers to a rounded point
# and curls up into a raised nose, like carved wooden klompen.
xs=np.array([-59,-58,-56,-53,-48,-44,-25,0,20,40,60,75,90,100,110,120,130,140,145,150,155,160,165])
width=PchipInterpolator(xs,[0,12,21,29,33.5,36,41,44,46.5,49,52,53.5,54,53.5,52,48.5,44,35,30,25,20,15,10])
roof=PchipInterpolator(xs,[48,50,51.5,53,54.5,55.5,57,57,58,60,66,69,69,66.5,67,65.5,62,56,53,50.5,49,47.5,46.5])
# Arch relief 7 mm deep so the sole under the arch stays >=10 mm. v7 makes it
# read like a real klomp: a steep heel-block front, a long flat arch and a
# crisp step up to the ball, instead of a shallow wave.
# Toe spring: nothing touches the ground ahead of the front pad land (x=135),
# so the sole sweeps up from there into a raised, upturned nose.
# Smooth toe spring: the robot's foot ends at x~133, so from the end of the
# front pad land the sole curves up in one long arc into the upturned tip,
# while the vamp line flows down from the crown and flattens into it.
bottom=PchipInterpolator([-59,-56,0,4,9,14,52,62,69,74,135,140,145,150,155,160,165],
                         [2,0,0,2.5,6.3,7,7,6,2.5,0,0,1.2,3.5,7,12,18.5,27])
# Rounded raised nose: width, roof and sole close onto the nose point with a
# square-root (elliptical) profile, so the tip is blunt-round rather than pinched.
TIP_START,TIP,NOSE_Z=165.,170.,43.5
def capped(f,end):
    def g(x):
        x=np.asarray(x,float);u=np.sqrt(np.clip((TIP-x)/(TIP-TIP_START),0,1))
        return np.where(x<TIP_START,f(np.minimum(x,TIP_START)),end+(f(TIP_START)-end)*u)
    return g
width,roof,bottom=capped(width,0.),capped(roof,NOSE_Z),capped(bottom,NOSE_Z)
TOE_LIFT=0.   # v10 shapes the upturn directly in the profiles above
def smooth(u):
    u=float(np.clip(u,0,1));return u*u*(3-2*u)
def SUPER_AT(x):
    # Superellipse crown: full at the heel, rounder (domed) over the toe cap.
    return 2.5-.35*smooth((x-80)/40)
sample=np.unique(np.r_[np.linspace(-59,TIP,320),xs,[-56,0,74,135],TIP-np.geomspace(.004,8,40)])
rings=[]
for x in sample:
    w=max(.02,float(width(x)));b=float(bottom(x));top=float(roof(x));h=top-b
    # Upturned point: ahead of the robot's toe (x>140) the whole snout bends
    # up, so the vamp line dips and then rises into a tip that points upward.
    lift=TOE_LIFT*max(0.,(x-140)/(TIP-140))**2;b+=lift;top+=lift
    # Ahead of the front pad land the flat sole narrows into a rounded keel,
    # so the raised nose is round underneath instead of a flat chin.
    t=float(np.clip((x-136)/(TIP-136),0,1));t=t*t*(3-2*t)
    f=.88-.7*t
    cheek1=(1-t)*min(1.8,.15*h)+t*.16*h;cheek2=(1-t)*min(6,.28*h)+t*.38*h
    # Rounder sole edges along the arch waist deepen its shadow from the side.
    a=smooth((x-8)/8)*smooth((66-x)/8);f-=.08*a;cheek1+=1.2*a;cheek2+=3.5*a
    SUPER=SUPER_AT(x)
    side=b+max(cheek2+.01,.36*h)
    # Broader ground lands retain a full border around the unscaled old pads.
    points=[[x,1-f*w,b],[x,1+f*w,b],
            [x,1+(f+(1-f)*.5)*w,b+cheek1],[x,1+(f+(1-f)*.875)*w,b+cheek2],[x,1+w,side]]
    for t in np.linspace(0,math.pi,73)[1:]:
        c,s=math.cos(t),math.sin(t)
        points.append([x,1+w*math.copysign(abs(c)**(2/SUPER),c),side+(top-side)*abs(s)**(2/SUPER)])
    points += [[x,1-(f+(1-f)*.875)*w,b+cheek2],[x,1-(f+(1-f)*.5)*w,b+cheek1]]
    rings.append(points)
rings=np.array(rings);n,k,_=rings.shape;verts=rings.reshape(-1,3).tolist();faces=[]
for i in range(n-1):
    for j in range(k):
        a=i*k+j;b=i*k+(j+1)%k;c=b+k;d=a+k;faces += [(a,b,c),(a,c,d)]
for i in [0,n-1]:
    c=len(verts);verts.append(rings[i].mean(0))
    for j in range(k):faces.append((c,i*k+j,i*k+(j+1)%k))
outer=trimesh.Trimesh(verts,faces);outer.fix_normals();outer=clean(outer,.02)
assert outer.is_volume

upper_zone=box([-100,-100,17],[220,100,400])

# Klomp opening: a U-shaped throat in plan view. Its apex sits where the
# toe-first donning path and the full-dorsiflexion shank clearance allow a
# closed vamp (about x=98), so the clearance cuts stay inside this smooth edge. The rim runs low along the
# sides and rises tangentially into the closed vamp, apex at x=THROAT on the
# centreline and swept back toward the sides.
THROAT=94.;THROAT_SWEEP=.0035;THROAT_RISE=28.
rim_base=PchipInterpolator([-120,-59,-50,-35,-15,10,30,50,70,90,110,240],
                           [43,43,43,42,41.5,42,45,49.5,55,60,63,63])
def rim(X,Y):
    u=np.clip((X-THROAT+THROAT_SWEEP*(Y-1)**2)/THROAT_RISE,0,None)
    return rim_base(X)+120*np.minimum(u,1.5)**2
opening=heightfield_cutter(np.arange(-100,221,1.),np.arange(-80,81,1.),rim)

# The closed vamp means the foot can no longer drop in from above. It goes in
# toe-first, as a person puts on a clog: heel raised DON_TILT about the toe,
# toe slid forward under the vamp along the tilted sole, then the heel lowered.
# The shin is kept upright meanwhile (ankle plantarflexed by DON_TILT, inside
# the +0.345 rad joint range), so leg and ankle cross only translate.
DON_TILT=math.radians(8);DON_PIVOT=np.array([ctx['pair_foot_hull'].bounds[1,0]+1,0,25.])
assert DON_TILT<.345
def don_poses():
    """(foot transform, leg/cross transform) samples from seated to fully out."""
    out=[]
    for t in np.linspace(0,DON_TILT,13):
        F=trimesh.transformations.rotation_matrix(t,[0,1,0],DON_PIVOT);out.append(F)
    direction=np.array([-math.cos(DON_TILT),0,math.sin(DON_TILT)])
    for s in np.r_[np.arange(5,120,5),[120,140,170,200]]:
        F=out[12].copy();F[:3,3]+=s*direction;out.append(F)
    poses=[]
    for F in out:
        L=np.eye(4);L[:3,3]=trimesh.transform_points([ctx['ankle_origin']],F)[0]-ctx['ankle_origin']
        poses.append((F,L))
    return poses
DON=don_poses()
def don_sweep(mesh,which):
    """Union of pairwise hulls along the donning path (exact for each translation segment)."""
    placed=[]
    for F,L in DON:
        m=mesh.copy();m.apply_transform(F if which=='foot' else L);placed.append(m)
    return union_md([hull_of(a,b) for a,b in zip(placed,placed[1:])])

print('Cutting toe-first donning sweep',flush=True)
foot_entry=union_md([I(don_sweep(offset_hull(p,1.01)[0],'foot'),upper_zone) for p in foot_pieces])
# The ankle cross follows the foot in pitch, so it tilts with it.
cross_clear,_=offset_hull(ctx['pair_cross_hull'],1.01)
cross_entry=I(don_sweep(cross_clear,'foot'),upper_zone)
leg_clear,_=offset_hull(ctx['leg_clip'],1.01)
leg_entry=I(don_sweep(leg_clear,'leg'),upper_zone)
cutters=[foot_entry,cross_entry,leg_entry,opening]
poses=[]
for pitch in ctx['PITCH_SAMPLES']:
    for roll in ctx['ROLL_SAMPLES']:
        leg=rotate_at(ctx['pair_shank_hull'],-pitch,[0,1,0],ctx['ankle_origin'])
        leg=rotate_at(leg,-roll,[1,0,0],ctx['ankle_origin'])
        poses.append((pitch,roll,leg))
        clipped=I(leg,box([-250,-250,15],[250,250,150]))
        if len(clipped.faces):
            expanded,_=offset_hull(clipped,1.01)
            cutters.append(I(expanded,upper_zone))
# Upper shell: hollow the upper (above the Z=17 foot rest) to UPPER_WALL so
# the toe cap and sides print as solid perimeters with no infill time. The
# sole below Z=17 stays solid.
UPPER_WALL=2.4
kernel=md.Manifold.sphere(1,12)
q=kernel.to_mesh();kernel_radius=UPPER_WALL/min(-ConvexHull(q.vert_properties[:,:3]).equations[:,3])
print('Hollowing upper',flush=True)
q=to_md(outer).minkowski_difference(kernel.scale([kernel_radius]*3)).simplify(1e-4).to_mesh64()
inner=trimesh.Trimesh(q.vert_properties[:,:3],q.tri_verts,process=False);assert inner.is_volume
inner=I(inner,upper_zone)
body=clean(D(outer,inner,*cutters))
for x,y in ctx['mounts']:
    body=clean(D(body,cyl(2.75,-10,17.1,x,y),cyl(4.6,-10,13,x,y)))
assert body.is_volume and len(body.split())==1

# Carved decoration, as on the reference clog: a six-petal rosette chip-carved
# into the vamp top and a scalloped line along each side. Grooves are 1.4 mm
# wide and 0.7 mm deep into the 3 mm shell, cut normal to the outer surface.
GROOVE_R=.7
def surface_points(path2d,direction):
    """Project plan (x,y) or side (x,z) points onto the outer envelope surface."""
    origins=[];dirs=[]
    for p in path2d:
        if direction=='down':origins.append([p[0],p[1],200]);dirs.append([0,0,-1])
        else:
            sign=1 if direction=='+y' else -1
            origins.append([p[0],1+sign*120,p[1]]);dirs.append([0,-sign,0])
    loc,ray,_=outer.ray.intersects_location(np.array(origins,float),np.array(dirs,float),multiple_hits=False)
    out=np.full((len(path2d),3),np.nan);out[ray]=loc
    return out
def groove(path2d,direction):
    pts=surface_points(path2d,direction)
    assert not np.isnan(pts).any(),'groove leaves the surface'
    ball=trimesh.creation.icosphere(1,GROOVE_R)
    segs=[]
    for a,b in zip(pts,pts[1:]):
        segs.append(hull_of(ball.copy().apply_translation(a),ball.copy().apply_translation(b)))
    return segs
def arc(cx,cy,r,a0,a1,step=1.):
    n=max(4,int(abs(a1-a0)*r/step))
    t=np.linspace(a0,a1,n);return np.c_[cx+r*np.cos(t),cy+r*np.sin(t)]
def rosette(cx,cy,R,direction):
    """Six-petal chip-carved rosette: two rings, six 120-degree petal arcs, tick chips."""
    out=groove(arc(cx,cy,R,0,2*math.pi),direction)+groove(arc(cx,cy,R+3.2,0,2*math.pi),direction)
    for i in range(6):
        a=i*math.pi/3;px,py=cx+R*math.cos(a),cy+R*math.sin(a)
        out+=groove(arc(px,py,R,a+2*math.pi/3,a+4*math.pi/3),direction)
    for i in range(24):
        a=i*2*math.pi/24;b=a+2*math.pi/48
        out+=groove(np.array([(cx+(R+.2)*math.cos(a),cy+(R+.2)*math.sin(a)),
                              (cx+(R+3)*math.cos(b),cy+(R+3)*math.sin(b))]),direction)
    return out
carve=[]
# Vamp top rosette, and a smaller one on each side of the toe cap (as on the reference).
carve+=rosette(124.,1.,10.,'down')
for direction in ['+y','-y']:carve+=rosette(119.,41.,7.5,direction)
# Wolf-tooth chip border across the vamp just ahead of the throat: two rails
# following the U throat with a zigzag between them.
ys=np.linspace(-44,46,91)
rail=lambda x0,y:x0-.0035*(y-1)**2
carve+=groove(np.c_[rail(104.5,ys),ys],'down')+groove(np.c_[rail(109.,ys),ys],'down')
zig=np.c_[[rail(104.5,y) if i%2==0 else rail(109.,y) for i,y in enumerate(np.linspace(-44,46,31))],np.linspace(-44,46,31)]
carve+=groove(zig,'down')
# Three diamond chips down the nose, from the rosette toward the tip.
for xc,r in [(141.,2.6),(147.,2.2),(152.,1.8)]:
    carve+=groove(np.array([(xc-r,1),(xc,1+r*.8),(xc+r,1),(xc,1-r*.8),(xc-r,1)]),'down')
# Scalloped line along each side, ending before the side rosettes.
side_x=np.linspace(-38,100,120)
side_z=PchipInterpolator([-38,0,40,80,100],[30,27,30,33,35])(side_x)+2.2*np.sin(side_x/9)
for direction in ['+y','-y']:carve+=groove(np.c_[side_x,side_z],direction)
carve=union_md(carve)
body=clean(D(body,carve))
# Drop zero-volume chips left where adjacent grooves meet.
parts=sorted(body.split(),key=lambda p:p.volume)
assert all(p.volume<1 for p in parts[:-1]),[p.volume for p in parts]
body=parts[-1]
assert body.is_volume and len(body.split())==1
print('One-piece klomp body complete',flush=True)

report={'units':'mm','style':'closed-toe klomp with carved rosette',
        'reference_photos':['reference/Latest_Clog_Reference.png','reference/reference_clogs.webp'],
        'one_piece_per_shoe':True,'upper_shell_wall_mm':UPPER_WALL,
        'cavity_allowance_mm':1,'cavity_cut_offset_mm':1.01,
        'foot_envelope':{'type':'union of convex hulls of x-slabs of both feet','x_cuts_mm':FOOT_CUTS},
        'throat_apex_x_mm':THROAT,'decor_groove_mm':{'width':2*GROOVE_R,'depth':GROOVE_R},
        'donning':{'method':'toe-first: heel raised about toe, slide along tilted sole',
                   'heel_tilt_deg':math.degrees(DON_TILT),'pivot_mm':DON_PIVOT.tolist(),
                   'shin_kept_upright':True},
        'mount_holes':{'diameter_mm':5.5,'underside_access_mm':9.2,'centers_xy_mm':ctx['mounts']},
        'extra_side_holes':0,'pad_recess_depth_mm':4,'pad_glue_gap_mm':.2,
        'pad_lateral_clearance_mm':.2,'rubber_ground_z_mm':-4.7,
        'existing_pad_scaling':1.0,'new_molds_generated':False,
        'physical_fit_tested':False,'feet':{}}
wood=[.87,.75,.53];rubber=[.12,.13,.14];pair=[]

def make_pad_and_pocket(hand,name):
    filename=name+'_Rubber_Insert'+('_Right' if name=='Front' and hand=='Right' else '')+'.step'
    source=HERE/'reference'/filename
    pad=cq.importers.importStep(str(source)).val()
    b=pad.BoundingBox()
    # Translation ONLY: retain every drafted wall and tread feature from the
    # user's existing mold. Right front uses its original handed source CAD.
    xmin=80 if name=='Front' else -48
    yc=1 if hand=='Left' else -1
    shift=[xmin-b.xmin,yc-(b.ymin+b.ymax)/2,3.8-b.zmax]
    pad=pad.translate(tuple(shift))
    face=max((f for f in pad.Faces() if f.geomType()=='PLANE' and f.normalAt().z>.99),key=lambda f:f.Area())
    top=face.outerWire()
    wider=max(top.offset2D(.2,'arc'),key=lambda w:cq.Face.makeFromWires(w).Area())
    floor_z=face.Center().z
    tool=cq.Solid.extrudeLinear(wider.translate((0,0,-12-floor_z)),[],cq.Vector(0,0,16))
    # Check the entire pocket boundary against the new flat contact footprint.
    a,z=(-56,0) if name=='Heel' else (74,135)
    x=np.linspace(a,z,180)
    ground=Polygon([(float(t),yc-.88*float(width(t))) for t in x]+
                   [(float(t),yc+.88*float(width(t))) for t in x[::-1]])
    pts=[]
    from shapely.geometry import Point
    for edge in wider.Edges():
        for t in np.linspace(0,1,max(3,int(edge.Length())+1)):
            p=edge.positionAt(float(t));pts.append([p.x,p.y])
    assert all(ground.covers(Point(p)) for p in pts),(hand,name,'pocket outside land')
    border=min(ground.boundary.distance(Point(p)) for p in pts)
    assert border>=3.9,(hand,name,border)
    dimensions=pad.BoundingBox()
    details={'source':filename,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'scale':1.,'translation_mm':shift,
             'size_mm':[dimensions.xlen,dimensions.ylen,dimensions.zlen],
             'minimum_ground_border_mm':border}
    return cqmesh(pad),cqmesh(tool),details

for hand in ['Left','Right']:
    shoe=body.copy()
    if hand=='Right':shoe.apply_transform(np.diag([1,-1,1,1]))
    pads={};pad_info={}
    for name in ['Front','Heel']:
        pad,pocket,details=make_pad_and_pocket(hand,name)
        shoe=clean(D(shoe,pocket));pads[name]=pad;pad_info[name]=details
    shoe=write_mesh(shoe,PRINT/f'K1_{hand}_Klomp_One_Piece.stl')
    normalized=shoe.copy()
    if hand=='Right':normalized.apply_transform(np.diag([1,-1,1,1]))
    foot=ctx['foot'] if hand=='Left' else ctx['right_foot_normalized']
    static=overlap(normalized,foot_env)
    upper=I(normalized,upper_zone)
    allowance=overlap(upper,cavity_env)
    assert static<.02 and allowance<.02,(hand,static,allowance)
    # Every sampled pose of the toe-first donning path, on the exported mesh.
    entry={}
    for i,(F,L) in enumerate(DON):
        values={}
        for name,T in [('foot_envelope',F),('pair_cross_hull',F),('leg_clip',L)]:
            moved=(foot_env if name=='foot_envelope' else ctx[name]).copy();moved.apply_transform(T)
            values[name]=overlap(normalized,moved)
        assert max(values.values())<.02,(hand,i,values)
        entry[str(i)]=values
    motion={}
    for pitch,roll,leg in poses:
        volume=overlap(normalized,leg)
        assert volume<.02,(hand,pitch,roll,volume)
        motion[f'{pitch},{roll}']=volume
    for name,pad in pads.items():
        value=overlap(shoe,pad);assert value<.02,(hand,name,value)
        pad_info[name]['shoe_overlap_mm3']=value
        # The old cast pad can enter from underneath without an undercut.
        release={}
        for dz in [0,-.5,-2,-4,-8]:
            m=pad.copy();m.apply_translation([0,0,dz]);v=overlap(shoe,m)
            assert v<.02,(hand,name,dz,v)
            release[str(dz)]=v
        pad_info[name]['pad_placement_overlap_mm3']=release
    report['feet'][hand]={'watertight':shoe.is_watertight,'single_solid':len(shoe.split())==1,
        'size_mm':shoe.extents.tolist(),'volume_mm3':shoe.volume,
        'static_foot_overlap_mm3':static,'upper_overlap_1mm_allowance_mm3':allowance,
        'toe_first_donning_overlap_mm3':entry,'sampled_ankle_overlap_mm3':motion,
        'existing_pads':pad_info}
    entries=[(shoe,wood,[0,0,0])]+[(p,rubber,[0,0,0]) for p in pads.values()]
    for m,col,_ in entries:pair.append((m,col,[0,-64 if hand=='Left' else 64,0]))
    # Do not export pad/mold print files: the user is reusing existing tooling.
    if hand=='Left':
        for name,elev,azim in [('Clog_Preview',22,-62),('Front_Three_Quarter',18,-25),('Side_Preview',0,-90),
                               ('Top_Preview',90,-90),('Underside_With_Existing_Pads',-35,-62)]:
            render(entries,elev,azim,(1500,1050)).save(HERE/(name+'.png'))
        render([(shoe,wood,[0,0,0])],-65,-65,(1500,1050)).save(HERE/'Existing_Pad_Pockets.png')
        cut=[(I(m,box([-100,1,-20],[220,100,160])),col,[0,0,0]) for m,col,_ in entries]
        render(cut+[(foot_env,[.3,.36,.42],[0,0,0])],15,-78,(1500,1050)).save(HERE/'Foot_Fit_Section.png')
        # Show the foot part-way along the toe-first path: heel up, toe entering the vamp.
        F,_=DON[18];tilted=foot_env.copy();tilted.apply_transform(F)
        render([(m,col,s) for m,col,s in cut]+[(tilted,[.3,.36,.42],[0,0,0])],0,-90,(1500,1050)).save(HERE/'Foot_Entry_Preview.png')
    print(hand,'one-piece STL, old-pad fit, toe-first donning and 35 poses passed',flush=True)
render(pair,40,-30,(1500,1200)).save(HERE/'Pair_Preview.png')
report['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PRINT.glob('*.stl')}
(HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',flush=True)
