import pickle
import helper_hpc as helper
import argparse
import numpy as np

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--input', help='pickle file input with metrics for plotting')
parser.add_argument('--ylabel', help='label for the y axis', default='Fitness')
parser.add_argument('--xlabel', help='label for the x axis', default='Generation')
args = parser.parse_args()

with open(args.input, 'rb') as f:
    pickled_metrics = pickle.load(f)
    print(pickled_metrics)

if type(pickled_metrics) == dict:
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x) for k, x in pickled_metrics.items()], name=[k for k,x in pickled_metrics.items()], x_label="Generation", y_label="Fitness", compute_CI=True)

    