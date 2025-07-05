import numpy as np
import vart
import xir
import os

# Load .xmodel
print("📦 Loading compiled XModel...")
graph = xir.Graph.deserialize("EfficientUNetBeamformer_int.xmodel")
subgraph = [s for s in graph.get_root_subgraph().toposort_child_subgraph() if s.has_attr("device")][0]
runner = vart.Runner.create_runner(subgraph, "run")

# Load input
input_path = os.path.join("board_test_data", "test_input_0.npy")
print(f"📂 Loading test input from {input_path}...")
input_tensor = np.load(input_path).astype(np.float32)

expected_shape = tuple(runner.get_input_tensors()[0].dims)
assert input_tensor.shape == expected_shape, \
    f"❌ Input shape {input_tensor.shape} does not match model input {expected_shape}"
input_data = [input_tensor]

# Prepare output buffer
output_dims = runner.get_output_tensors()[0].dims
output_data = [np.empty(output_dims, dtype=np.float32)]

# Run inference
print("🚀 Running inference on DPU...")
job_id = runner.execute_async(input_data, output_data)
runner.wait(job_id)

# Save prediction
output_path = os.path.join("board_test_data", "output_0.npy")
np.save(output_path, output_data[0])
print(f"✅ Inference complete. Output saved to {output_path}")