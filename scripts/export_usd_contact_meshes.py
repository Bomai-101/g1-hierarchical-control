"""Export enabled authored USD collider meshes in their rigid-body frames."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(str(args.asset));cache=UsdGeom.XformCache();meshes=[]
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
        if not prim.IsA(UsdGeom.Mesh):raise RuntimeError("Expected original collider mesh")
        owner=prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
        if not owner:raise RuntimeError("Collider has no rigid body")
        mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        matrix=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(owner).GetInverse()
        points=(np.c_[points,np.ones(len(points))]@np.asarray(matrix))[:,:3]
        faces=[];indices=list(mesh.GetFaceVertexIndicesAttr().Get());cursor=0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon=indices[cursor:cursor+count];faces.extend((polygon[0],polygon[i],polygon[i+1]) for i in range(1,count-1));cursor+=count
        file=args.output/f"contact_{owner.GetName()}.obj"
        with file.open("w") as stream:
            for point in points:stream.write("v "+" ".join(f"{x:.17g}" for x in point)+"\n")
            for face in faces:stream.write("f "+" ".join(str(x+1) for x in face)+"\n")
        meshes.append(dict(body=owner.GetName(),usd_path=str(prim.GetPath()),file=str(file.resolve()),vertices=points.tolist(),
            approximation=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    if {m["body"] for m in meshes}!={"left_ankle_roll_link","right_ankle_roll_link","torso_link"}:raise RuntimeError("Unexpected collider set")
    (args.output/"contacts.json").write_text(json.dumps(dict(meshes=meshes,asset=str(args.asset.resolve()),asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print("CONTACT_MESHES",len(meshes))


if __name__=="__main__":main()
