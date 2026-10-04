"""Read authored USD joint frames, masses and collider geometry without simulation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--headless",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        import numpy as np
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        stage=Usd.Stage.Open(str(args.asset))
        cache=UsdGeom.XformCache()
        prims=list(Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
        rigid={p.GetName():p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        base=rigid["pelvis"]
        inverse=cache.GetLocalToWorldTransform(base).GetInverse()
        def plain(value):
            if isinstance(value,float) and not math.isfinite(value):return str(value)
            if value is None or isinstance(value,(bool,int,float,str)):return value
            if hasattr(value,"GetReal"):return [float(value.GetReal()),*map(float,value.GetImaginary())]
            try:return [plain(x) for x in value]
            except TypeError:return str(value)
        def point(matrix,vector):return list(map(float,matrix.Transform(Gf.Vec3d(*vector))))
        def direction(matrix,vector):return list(map(float,matrix.TransformDir(Gf.Vec3d(*vector))))
        bodies=[]
        for name,prim in rigid.items():
            transform=cache.GetLocalToWorldTransform(prim)*inverse
            mass=UsdPhysics.MassAPI(prim)
            bodies.append(dict(name=name,path=str(prim.GetPath()),position=point(transform,[0,0,0]),
                rotation_rows=np.asarray(transform)[:3,:3].tolist(),mass=plain(mass.GetMassAttr().Get()),
                com_local=plain(mass.GetCenterOfMassAttr().Get()),diagonal_inertia=plain(mass.GetDiagonalInertiaAttr().Get()),
                principal_axes=plain(mass.GetPrincipalAxesAttr().Get())))
        joints=[]
        for prim in prims:
            if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)):continue
            revolute=prim.IsA(UsdPhysics.RevoluteJoint)
            joint=UsdPhysics.RevoluteJoint(prim) if revolute else UsdPhysics.FixedJoint(prim)
            paths=[joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()]
            frames=[]
            axis=str(joint.GetAxisAttr().Get()) if revolute else "X"
            unit={"X":[1,0,0],"Y":[0,1,0],"Z":[0,0,1]}[axis]
            for side in (0,1):
                if len(paths[side])!=1:raise RuntimeError("Expected two rigid body joint endpoints")
                body=stage.GetPrimAtPath(paths[side][0])
                transform=cache.GetLocalToWorldTransform(body)*inverse
                pos=getattr(joint,f"GetLocalPos{side}Attr")().Get()
                quat=getattr(joint,f"GetLocalRot{side}Attr")().Get()
                rotated=Gf.Rotation(Gf.Quatd(float(quat.GetReal()),Gf.Vec3d(*quat.GetImaginary()))).TransformDir(Gf.Vec3d(*unit))
                rotation=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(float(quat.GetReal()),Gf.Vec3d(*quat.GetImaginary()))))*transform
                frames.append(dict(body=body.GetName(),local_pos=plain(pos),local_quat=plain(quat),
                    anchor=point(transform,pos),axis=direction(transform,rotated),frame_rotation_rows=np.asarray(rotation)[:3,:3].tolist()))
            joints.append(dict(name=prim.GetName(),path=str(prim.GetPath()),kind="revolute" if revolute else "fixed",axis_token=axis,frames=frames,
                lower_deg=plain(joint.GetLowerLimitAttr().Get()) if revolute else None,upper_deg=plain(joint.GetUpperLimitAttr().Get()) if revolute else None))
        colliders=[]
        for prim in prims:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            owner=prim
            while owner and owner.GetName() not in rigid:owner=owner.GetParent()
            if not owner:continue
            body=owner.GetName()
            transform=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(owner).GetInverse()
            vertices=None
            if prim.IsA(UsdGeom.Mesh):vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            elif prim.IsA(UsdGeom.Cube):
                size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                vertices=np.array([[x,y,z] for x in (-size/2,size/2) for y in (-size/2,size/2) for z in (-size/2,size/2)])
            elif prim.HasAttribute("extent"):
                extent=prim.GetAttribute("extent").Get()
                if extent is not None:
                    vertices=np.array([[x,y,z] for x in (extent[0][0],extent[1][0]) for y in (extent[0][1],extent[1][1]) for z in (extent[0][2],extent[1][2])])
            local=(np.c_[vertices,np.ones(len(vertices))]@np.asarray(transform))[:,:3] if vertices is not None else None
            attrs={a.GetName():plain(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(("physics:","physxCollision:","physxMeshCollision:"))}
            colliders.append(dict(path=str(prim.GetPath()),body=body,type=prim.GetTypeName(),enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                bounds_local=None if local is None else [local.min(axis=0).tolist(),local.max(axis=0).tolist()],
                vertices_local=local.tolist() if local is not None and "ankle_roll" in body else None,attributes=attrs))
        result=dict(asset=str(args.asset.resolve()),asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),
            meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),up_axis=str(UsdGeom.GetStageUpAxis(stage)),
            coordinate_definition="Authored USD transforms relative to pelvis; not simulated state or inferred contact hull",bodies=bodies,joints=joints,colliders=colliders)
        (args.output/"usd_geometry.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        (args.output/"source.json").write_text(json.dumps({str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
        print("USD_GEOMETRY",len(bodies),len(joints),len(colliders),flush=True)
    finally:pass


if __name__=="__main__":main()
