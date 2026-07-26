"""Rebuild NAF pickle from npz using numpy 1.x (run with aet4dgs env)."""
import numpy as np
import pickle
from pathlib import Path
import sys

def rebuild(data_dir):
    data_dir = Path(data_dir)
    d = np.load(data_dir / "data_arrays.npz")
    data = {
        "DSD": float(d["DSD"]), "DSO": float(d["DSO"]),
        "nVoxel": d["nVoxel"].tolist(), "dVoxel": d["dVoxel"].tolist(),
        "nDetector": d["nDetector"].tolist(), "dDetector": d["dDetector"].tolist(),
        "offOrigin": d["offOrigin"].tolist(), "offDetector": d["offDetector"].tolist(),
        "totalAngle": float(d["totalAngle"]), "startAngle": float(d["startAngle"]),
        "accuracy": float(d["accuracy"]), "mode": "parallel",
        "numTrain": int(d["numTrain"]), "numVal": int(d["numVal"]),
        "image": np.array(d["image"]),
        "train": {k: np.array(d[f"train_{k}"]) for k in ["angles","projections","time","phase"]},
        "val": {k: np.array(d[f"val_{k}"]) for k in ["angles","projections","time","phase"]},
    }
    with open(data_dir / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)
    # Also rebuild init_data.npy with this numpy
    n_points = 50000
    init = np.concatenate([(np.random.rand(n_points, 3) - 0.5) * 2,
                           np.random.rand(n_points, 1) * 0.5], axis=1).astype(np.float32)
    np.save(data_dir / "init_data.npy", init)
    print(f"Rebuilt {data_dir.name}: image={data['image'].shape}")

if __name__ == "__main__":
    structs = sys.argv[1:] if len(sys.argv) > 1 else ["Cu-309", "Cu-923", "Pt-923", "FePt-923"]
    base = Path("/data/aet-bench/data/simulated")
    for s in structs:
        rebuild(base / s)
