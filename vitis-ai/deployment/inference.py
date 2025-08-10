import numpy as np
import vart
import xir
import os
import time  # ⏱️ Add time module

input_scale = 1.0 / 128
input_zero_point = 0

print("📦 Loading compiled XModel...")
graph = xir.Graph.deserialize("CustomEfficientUNet.xmodel")

subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"]
assert len(dpu_subgraph) == 1, "❌ Expected exactly one DPU subgraph"
runner = vart.Runner.create_runner(dpu_subgraph[0], "run")

data_dir = "board_test_data"

# ⏱️ Track total inference time
total_start_time = time.time()

for i in range(6):
    input_file = os.path.join(data_dir, f"test_input_{i}.npy")
    output_file = os.path.join(data_dir, f"output_{i}.npy")

    print(f"\n📂 Loading input: {input_file}")
    input_tensor = np.load(input_file).astype(np.float32)
    input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))

    quant_input = np.clip(np.round(input_tensor / input_scale + input_zero_point), -128, 127).astype(np.int8)

    expected_shape = tuple(runner.get_input_tensors()[0].dims)
    assert quant_input.shape == expected_shape, f"❌ Shape mismatch: {quant_input.shape} vs expected {expected_shape}"

    input_data = [quant_input]
    output_tensor = runner.get_output_tensors()[0]
    output_data = [np.empty(tuple(output_tensor.dims), dtype=np.int8)]

    print("🚀 Running inference...")
    
    # ⏱️ Per-inference timing
    start_time = time.time()
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    end_time = time.time()
    elapsed = (end_time - start_time) * 1000  # ms
    print(f"⏱️ Inference time: {elapsed:.2f} ms")

    dequant_output = (output_data[0].astype(np.float32) - input_zero_point) * input_scale
    np.save(output_file, dequant_output)
    print(f"✅ Output saved: {output_file}")

# ⏱️ Total time
total_elapsed = (time.time() - total_start_time) * 1000
print(f"\n🏁 All inferences complete in {total_elapsed:.2f} ms")
