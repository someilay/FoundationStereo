"""
Count FP32/FP16 initializers (weights) and show graph I/O types in an ONNX model.

Usage:
    python scripts/onnx_check_dtype.py --onnx_path pretrained_models/foundation_stereo_1888_1056.onnx
"""
import argparse
import onnx


def dtype_name(data_type: int) -> str:
    return onnx.TensorProto.DataType.Name(data_type)


def count_dtypes(onnx_path: str) -> None:
    model = onnx.load(onnx_path)

    # --- initializers (weights) ---
    fp32_count = 0
    fp16_count = 0
    other: dict = {}

    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT:
            fp32_count += 1
        elif init.data_type == onnx.TensorProto.FLOAT16:
            fp16_count += 1
        else:
            name = dtype_name(init.data_type)
            other[name] = other.get(name, 0) + 1

    total = fp32_count + fp16_count + sum(other.values())
    print(f"Model: {onnx_path}")
    print(f"  FP32 initializers : {fp32_count}")
    print(f"  FP16 initializers : {fp16_count}")
    for name, count in other.items():
        print(f"  {name:18s}: {count}")
    print(f"  Total initializers: {total}")

    # --- graph inputs / outputs (determines Triton config data_type) ---
    print()
    print("  Graph inputs:")
    for t in model.graph.input:
        elem_type = t.type.tensor_type.elem_type
        shape = [
            (d.dim_value if d.dim_value > 0 else d.dim_param)
            for d in t.type.tensor_type.shape.dim
        ]
        print(f"    {t.name:10s}  {dtype_name(elem_type):8s}  {shape}")

    print("  Graph outputs:")
    for t in model.graph.output:
        elem_type = t.type.tensor_type.elem_type
        shape = [
            (d.dim_value if d.dim_value > 0 else d.dim_param)
            for d in t.type.tensor_type.shape.dim
        ]
        print(f"    {t.name:10s}  {dtype_name(elem_type):8s}  {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model file")
    args = parser.parse_args()
    count_dtypes(args.onnx_path)
