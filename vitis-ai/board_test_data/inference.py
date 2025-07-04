import numpy as np
import vart
import xir
from skimage.metrics import structural_similarity as ssim

# Load .xmodel
graph = xir.Graph.deserialize("EfficientUNetBeamformer_int.xmodel")
subgraph = [s for s in graph.get_root_subgraph().toposort_child_subgraph() if s.has_attr("device")][0]
runner = vart.Runner.create_runner(subgraph, "run")

# Load input
input_tensor = np.load("test_input_0.npy").astype(np.float32)
assert input_tensor.shape == tuple(runner.get_input_tensors()[0].dims), \
    f"Input shape {input_tensor.shape} does not match model input {runner.get_input_tensors()[0].dims}"
input_data = [input_tensor]

# Prepare output buffer
output_dims = runner.get_output_tensors()[0].dims
output_data = [np.empty(output_dims, dtype=np.float32)]

# Run inference
job_id = runner.execute_async(input_data, output_data)
runner.wait(job_id)

# Save prediction
pred = output_data[0]
np.save("output_0.npy", pred)

# Load GT and compute SSIM
gt = np.load("test_gt_0.npy").squeeze()  # shape: [1,1,192,304] -> [192,304]
out = pred.squeeze()                     # shape: [1,1,192,304] -> [192,304]
ssim_score = ssim(gt, out, data_range=gt.max() - gt.min())
print(f"SSIM with ground truth: {ssim_score:.4f}")