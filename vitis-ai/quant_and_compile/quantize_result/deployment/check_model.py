import xir

graph = xir.Graph.deserialize("CustomEfficientUNet.xmodel")
subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

print("Subgraphs:")
for sg in subgraphs:
    print(f"  Name: {sg.get_name()}")
    if sg.has_attr("device"):
        print(f"    Device: {sg.get_attr('device')}")