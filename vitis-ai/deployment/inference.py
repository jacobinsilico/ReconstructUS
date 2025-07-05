import numpy as np
import vart
import xir
import os

# 🚨 Constants — these MUST match quantization (adjust if needed)
input_scale = 1.0 / 128  # Assumed int8 range [-128, 127] mapped from [-1, 1]
input_zero_point = 0     # Often 0 when symmetric quantization is used

# 📦 Load compiled XModel
print("📦 Loading compiled XModel...")
graph = xir.Graph.deserialize("CustomEfficientUNet.xmodel")

# 🧠 Find the subgraph that runs on DPU
subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"]
assert len(dpu_subgraph) == 1, "❌ Expected exactly one DPU subgraph"
runner = vart.Runner.create_runner(dpu_subgraph[0], "run")

# 📁 Input/output folder
data_dir = "board_test_data"

# 🔁 Loop over all input files
for i in range(6):
    input_file = os.path.join(data_dir, f"test_input_{i}.npy")
    output_file = os.path.join(data_dir, f"output_{i}.npy")

    print(f"\n📂 Loading input: {input_file}")
    input_tensor = np.load(input_file).astype(np.float32)
    input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))

    # 🧮 Quantize input (float32 [-1,1] → int8 [-128,127])
    quant_input = np.clip(np.round(input_tensor / input_scale + input_zero_point), -128, 127).astype(np.int8)

    # 🎯 Shape check
    expected_shape = tuple(runner.get_input_tensors()[0].dims)
    assert quant_input.shape == expected_shape, f"❌ Shape mismatch: {quant_input.shape} vs expected {expected_shape}"

    input_data = [quant_input]
    output_tensor = runner.get_output_tensors()[0]
    output_data = [np.empty(tuple(output_tensor.dims), dtype=np.int8)]

    print("🚀 Running inference...")
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)

    # 🧼 Dequantize output (int8 → float32)
    dequant_output = (output_data[0].astype(np.float32) - input_zero_point) * input_scale

    np.save(output_file, dequant_output)
    print(f"✅ Output saved: {output_file}")

print("\n🏁 All inferences complete.")
