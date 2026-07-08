"""
Convert an FP32 ONNX model to FP16.

Uses onnxconverter_common which has no PyTorch dependency.
Install: pip install onnxconverter-common

Usage:
    python scripts/onnx_to_fp16.py \
        --input  pretrained_models/foundation_stereo_1888_1056.onnx \
        --output pretrained_models/foundation_stereo_1888_1056_fp16.onnx
"""
import argparse
import os
import onnx
from onnxconverter_common import float16


def convert(input_path: str, output_path: str) -> None:
    print(f"Loading  : {input_path}")
    model_fp32 = onnx.load(input_path)

    print("Converting to FP16 ...")
    # Block Cast ops from conversion: DINOv2 uses explicit Cast-to-float32 nodes
    # internally; converting them to float16 breaks graph type consistency.
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=False,
        op_block_list=['Cast'],
    )
    model_fp16 = onnx.shape_inference.infer_shapes(model_fp16)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model_fp16, output_path)
    print(f"Saved    : {output_path}")

    # quick summary
    fp32 = sum(1 for i in model_fp16.graph.initializer if i.data_type == onnx.TensorProto.FLOAT)
    fp16 = sum(1 for i in model_fp16.graph.initializer if i.data_type == onnx.TensorProto.FLOAT16)
    print(f"  FP32 initializers remaining : {fp32}")
    print(f"  FP16 initializers           : {fp16}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True,  help="Path to the input FP32 ONNX model")
    parser.add_argument("--output", type=str, required=True,  help="Path to save the FP16 ONNX model")
    args = parser.parse_args()
    convert(args.input, args.output)
