"""
Run FoundationStereo inference with a TensorRT engine (.plan / .engine) or ONNX model.

Usage (TensorRT):
    python scripts/run_demo_tensorrt.py \
        --left_img  assets/left.png \
        --right_img assets/right.png \
        --pretrained pretrained_models/foundation_stereo_1888_1056_fp16.plan \
        --height 1056 --width 1888

Usage (ONNX):
    python scripts/run_demo_tensorrt.py \
        --left_img  assets/left.png \
        --right_img assets/right.png \
        --pretrained pretrained_models/foundation_stereo_1888_1056.onnx \
        --height 1056 --width 1888
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import argparse
import logging
import os
import sys
import time
from typing import List

import cv2
import imageio
import numpy as np
import onnxruntime as ort
import open3d as o3d
import tensorrt as trt
import torch

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from Utils import depth2xyzmap, set_seed, toOpen3dCloud, vis_disparity


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(image_path: str, args) -> tuple:
    """Read an image, resize to (args.height x args.width), return (NCHW float32 tensor, HWC numpy)."""
    img = imageio.imread(image_path)
    if img.ndim < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if args.height and args.width:
        img = cv2.resize(img, (args.width, args.height))
    tensor = torch.as_tensor(img.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    return tensor, img


# ---------------------------------------------------------------------------
# TensorRT engine wrapper
# ---------------------------------------------------------------------------


class TrtEngine:
    """
    Thin synchronous wrapper around a TensorRT engine.

    Compatible with TensorRT 8+ (uses set_tensor_address / execute_async_v3).
    Uses torch CUDA tensors for zero-copy GPU buffer management.
    """

    def __init__(self, plan_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(plan_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logging.info(
            "TRT engine loaded: inputs=%s  outputs=%s",
            self.input_names,
            self.output_names,
        )

    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        stream = torch.cuda.current_stream().cuda_stream

        # Upload inputs, set shapes (required for dynamic-batch engines), bind addresses
        gpu_inputs: List[torch.Tensor] = []
        for name, arr in zip(self.input_names, inputs):
            t = torch.from_numpy(arr).cuda().contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            gpu_inputs.append(t)

        # Allocate output buffers and bind addresses
        gpu_outputs: List[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = (
                torch.float16 if trt_dtype == trt.DataType.HALF else torch.float32
            )
            t = torch.empty(shape, dtype=torch_dtype, device="cuda")
            self.context.set_tensor_address(name, t.data_ptr())
            gpu_outputs.append(t)

        self.context.execute_async_v3(stream)
        torch.cuda.synchronize()

        # Always return float32 numpy arrays
        return [t.float().cpu().numpy() for t in gpu_outputs]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def get_onnx_model(args) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        args.pretrained,
        sess_options=session_options,
        providers=["CUDAExecutionProvider"],
    )


def get_engine_model(args) -> TrtEngine:
    return TrtEngine(args.pretrained)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def inference(left_img_path: str, right_img_path: str, model, args) -> None:
    left_img, input_left = preprocess(left_img_path, args)
    right_img, _ = preprocess(right_img_path, args)

    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.time()

        if isinstance(model, ort.InferenceSession):
            left_disp = model.run(
                None,
                {"left": left_img.numpy(), "right": right_img.numpy()},
            )[0]
        else:
            left_disp = model.run([left_img.numpy(), right_img.numpy()])[0]

        torch.cuda.synchronize()
        logging.info("Inference time: %.3f s", time.time() - t0)

    left_disp = left_disp.squeeze()  # H×W

    os.makedirs(os.path.join(args.save_path, "visual"), exist_ok=True)
    vis = vis_disparity(left_disp)
    vis = np.concatenate([input_left, vis], axis=1)
    imageio.imwrite(
        os.path.join(args.save_path, "visual", os.path.basename(left_img_path)), vis
    )

    if args.pc:
        save_name = os.path.basename(left_img_path).rsplit(".", 1)[0] + ".ply"
        baseline = 193.001 / 1e3
        K = np.array(
            [1998.842, 0, 588.364, 0, 1998.842, 505.864, 0, 0, 1], dtype=np.float64
        ).reshape(3, 3)
        depth = K[0, 0] * baseline / (left_disp.astype(np.float64) + 1e-6)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), input_left.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
            np.asarray(pcd.points)[:, 2] <= args.z_far
        )
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        os.makedirs(os.path.join(args.save_path, "cloud"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(args.save_path, "cloud", save_name), pcd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FoundationStereo TensorRT / ONNX demo"
    )
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument("--left_img", "-l", required=True, help="Path to left image")
    parser.add_argument("--right_img", "-r", required=True, help="Path to right image")
    parser.add_argument(
        "--save_path",
        "-s",
        default=f"{code_dir}/../output",
        help="Directory to save results",
    )
    parser.add_argument(
        "--pretrained",
        default=f"{code_dir}/../pretrained_models/foundation_stereo_1888_1056_fp16.plan",
        help="Path to .plan / .engine (TensorRT) or .onnx model",
    )
    parser.add_argument("--height", type=int, default=1056, help="Input image height")
    parser.add_argument("--width", type=int, default=1888, help="Input image width")
    parser.add_argument("--pc", action="store_true", help="Save point cloud")
    parser.add_argument(
        "--z_far", default=100, type=float, help="Max depth to clip in point cloud"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    assert os.path.isfile(args.pretrained), f"Model not found: {args.pretrained}"
    logging.info("Loading model from %s", args.pretrained)
    set_seed(0)

    if args.pretrained.endswith(".onnx"):
        model = get_onnx_model(args)
    elif args.pretrained.endswith((".plan", ".engine")):
        model = get_engine_model(args)
    else:
        raise ValueError(
            f"Unknown model format: {args.pretrained}. Expected .onnx, .plan, or .engine"
        )

    inference(args.left_img, args.right_img, model, args)


if __name__ == "__main__":
    main()
