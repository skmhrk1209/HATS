import os
import argparse
from tensorflow.python.tools import freeze_graph

parser = argparse.ArgumentParser()
parser.add_argument("--input_graph", type=str, help="input_graph")
parser.add_argument("--input_checkpoint", type=str, help="input_checkpoint")
parser.add_argument("--output_node_names", type=str, help="output_node_names")
parser.add_argument("--output_graph", type=str, help="output_graph")
args = parser.parse_args()

freeze_graph.freeze_graph(
    input_graph=args.input_graph, 
    input_checkpoint=args.input_checkpoint,
    output_node_names=args.output_node_names,
    output_graph=args.output_graph,
    input_saver="", 
    input_binary=False,
    restore_op_name="",
    filename_tensor_name="",
    clear_devices=True,
    initializer_nodes=""
)
