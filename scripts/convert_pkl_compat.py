"""Convert NAF pickles to numpy 1.x compatible format via npz intermediate."""
import numpy as np
import pickle
from pathlib import Path
import sys

def convert(data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / "data.pkl", "rb") as f:
        data = pickle.load(f)

    arrays = {}
    for k in ["DSD", "DSO", "totalAngle", "startAngle", "accuracy", "numTrain", "numVal"]:
        arrays[k] = np.array(data[k])
    for k in ["nVoxel", "dVoxel", "nDetector", "dDetector", "offOrigin", "offDetector"]:
        arrays[k] = np.array(data[k])
    arrays["image"] = data["image"]
    for split in ["train", "val"]:
        for key in ["angles", "projections", "time", "phase"]:
            arrays[f"{split}_{key}"] = data[split][key]
    np.savez(data_dir / "data_arrays.npz", **arrays)
    print(f"Exported {data_dir.name}")

if __name__ == "__main__":
    structs = sys.argv[1:] if len(sys.argv) > 1 else ["Cu-309", "Cu-923", "Pt-923", "FePt-923"]
    base = Path("/data/aet-bench/data/simulated")
    for s in structs:
        convert(base / s)
