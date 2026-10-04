"""Plot verified default-pose and foot-collision differences from archived JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
import numpy as np


def hull(points):
    points=sorted(set(tuple(p) for p in points))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(values):
        result=[]
        for p in values:
            while len(result)>=2 and cross(result[-2],result[-1],p)<=0:result.pop()
            result.append(p)
        return result
    return half(points)[:-1]+half(list(reversed(points)))[:-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--results",type=Path,required=True);args=parser.parse_args()
    poses=json.loads((args.results/"plot_data.json").read_text());feet=json.loads((args.results/"feet.json").read_text())["left"]
    figure,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    for label,key,color in [("Isaac USD","usd_default_positions","#2374ab"),("MuJoCo","mujoco_default_positions","#dd7034")]:
        p=np.array(poses[key])*1000;axes[0].plot(p[:,0],p[:,2],"o-",label=label,color=color)
    axes[0].set(title="Same joint values, different leg geometry",xlabel="Forward x relative to pelvis (mm)",ylabel="Vertical z relative to pelvis (mm)",aspect="equal")
    axes[0].legend();axes[0].grid(alpha=.25)
    points=np.concatenate([np.array(c["vertices_local"])[:,:2] for c in feet["usd_colliders"]])*1000
    axes[1].add_patch(Polygon(hull(points),closed=True,facecolor="#2374ab",alpha=.25,edgecolor="#2374ab",label="Isaac convex-hull footprint"))
    for i,g in enumerate(feet["mujoco_spheres"]):
        axes[1].add_patch(Circle(np.array(g["center"])[:2]*1000,g["radius"]*1000,facecolor="#dd7034",label="MuJoCo sphere projections" if i==0 else None))
    axes[1].set(title="Foot collision geometry (body-local top view)",xlabel="Foot-local x (mm)",ylabel="Foot-local y (mm)",aspect="equal",xlim=(-80,150),ylim=(-65,65))
    axes[1].legend(loc="lower center",fontsize=9);axes[1].grid(alpha=.25)
    figure.suptitle("G1 model geometry audit | positions are kinematic, not a rollout",fontsize=13)
    figure.savefig(args.results/"geometry_comparison.png",dpi=160);figure.savefig(args.results/"geometry_comparison.svg")


if __name__=="__main__":main()
