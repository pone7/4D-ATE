"""Quick test: can X2-Gaussian load our Cu nanoparticle data?"""
import sys
sys.path.append(".")
from x2_gaussian.arguments import ModelParams, PipelineParams, ModelHiddenParams
from x2_gaussian.dataset import Scene
from argparse import ArgumentParser

parser = ArgumentParser()
lp = ModelParams(parser)
pp = PipelineParams(parser)
hp = ModelHiddenParams(parser)
args = parser.parse_args([
    "--source_path", "/data/aet-bench/data/cu_nano/data.pkl",
    "--model_path", "/data/aet-bench/output/cu_nano_test",
])

dataset = lp.extract(args)
print("Loading scene...")
scene = Scene(dataset, shuffle=False)
print(f"Train cameras: {len(scene.getTrainCameras())}")
print(f"Test cameras: {len(scene.getTestCameras())}")
mode = scene.scanner_cfg["mode"]
svoxel = scene.scanner_cfg["sVoxel"]
print(f"Scanner cfg mode: {mode}")
print(f"Scanner sVoxel: {svoxel}")
print(f"Vol GT shape: {scene.vol_gt.shape}")
print(f"Scene scale: {scene.scene_scale}")

cam = scene.getTrainCameras()[0]
print(f"Camera 0: angle={cam.angle:.3f}, mode={cam.mode}, time={cam.time}, phase={cam.phase}")
print(f"  Image shape: {cam.original_image.shape}, range=[{cam.original_image.min():.2f}, {cam.original_image.max():.2f}]")
print("Data loading OK!")
