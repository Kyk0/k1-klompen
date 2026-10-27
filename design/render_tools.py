from pathlib import Path
import numpy as np,trimesh
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def render(entries,elev=40,azim=-60,size=(1050,750)):
 tris=[];colors=[];norm=[]
 for m,col,shift in entries:
  tris.append(m.triangles+np.array(shift));norm.append(m.face_normals)
  if np.ndim(col)==1:col=np.tile(col,(len(m.faces),1))
  colors.append(col)
 tri=np.concatenate(tris);cols=np.concatenate(colors);ns=np.concatenate(norm)
 e=np.deg2rad(elev);a=np.deg2rad(azim)
 view=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
 right=np.array([-np.sin(a),np.cos(a),0]);up=np.cross(view,right)
 rot=np.stack([right,up,view],axis=1);p=tri@rot
 lo=p.reshape(-1,3).min(0);hi=p.reshape(-1,3).max(0);W,H=size;scale=min((W-100)/(hi[0]-lo[0]),(H-100)/(hi[1]-lo[1]));cen=(lo+hi)/2
 p[:,:,0]=(p[:,:,0]-cen[0])*scale+W/2;p[:,:,1]=H/2-(p[:,:,1]-cen[1])*scale
 bg=np.array([244,246,248],dtype=float);buf=np.tile(bg,(H,W,1));depth=np.full((H,W),-np.inf)
 light=(view*.7+up*.6+right*.3);light/=np.linalg.norm(light)
 lum=.40+.60*np.maximum(0,ns@light);color=np.clip(cols*lum[:,None]*255,0,255)
 # Two-sided presentation lighting for underneath views; mesh winding remains physical.
 for i,t in enumerate(p):
  x0=max(0,int(np.floor(t[:,0].min())));x1=min(W-1,int(np.ceil(t[:,0].max())));y0=max(0,int(np.floor(t[:,1].min())));y1=min(H-1,int(np.ceil(t[:,1].max())))
  if x1<x0 or y1<y0:continue
  den=(t[1,1]-t[2,1])*(t[0,0]-t[2,0])+(t[2,0]-t[1,0])*(t[0,1]-t[2,1])
  if abs(den)<1e-9:continue
  xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
  w0=((t[1,1]-t[2,1])*(xx-t[2,0])+(t[2,0]-t[1,0])*(yy-t[2,1]))/den
  w1=((t[2,1]-t[0,1])*(xx-t[2,0])+(t[0,0]-t[2,0])*(yy-t[2,1]))/den;w2=1-w0-w1
  z=w0*t[0,2]+w1*t[1,2]+w2*t[2,2];sub=depth[y0:y1+1,x0:x1+1]
  take=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)&(z>sub)
  sub[take]=z[take];buf[y0:y1+1,x0:x1+1][take]=color[i]
 return Image.fromarray(np.uint8(buf))



def chelsea_panels(entries):
 # Partition the display surfaces so the elastic panels never overlap the upper.
 result=entries[:2]
 for mesh,color,shift in entries[2:4]:
  remaining=[mesh]
  for side in [-1,1]:
   lo=[-30,side*36-5,65];hi=[20,side*36+7,116]
   outside=[]
   for original in remaining:
    inside=original
    for axis in range(3):
     for limit,sign in [(lo[axis],1),(hi[axis],-1)]:
      point=np.zeros(3);point[axis]=limit
      normal=np.zeros(3);normal[axis]=sign
      cut=inside.slice_plane(point,-normal)
      if len(cut.faces):outside.append(cut)
      inside=inside.slice_plane(point,normal)
    if len(inside.faces):result.append((inside,[.08,.085,.09],shift))
   remaining=outside
  result.extend((m,color,shift) for m in remaining if len(m.faces))
 return result
