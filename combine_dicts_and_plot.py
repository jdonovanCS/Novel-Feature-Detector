import helper_hpc as helper
import pickle
import numpy as np
import argparse

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--experiment_name', help='experiment names for accessing data (separate with comma and space) (ie. exp1, exp2)', default='')
parser.add_argument('--fixed_conv', help="Get data related to when the convolutional layers are fixed", default=False)
parser.add_argument('--plot_label', help="For each experiment supply a plot label (separate with command and space as in experiment_name", default='')
args = parser.parse_args()

if args.experiment_name == '':
    print("Please supply experiment name")
    exit()

experiment_names = args.experiment_name.split(', ')
if args.plot_label != '':
    plot_labels = args.plot_labels.split(', ')
else:
    plot_labels = experiment_names
fixed_conv = args.fixed_conv
name_add = ''
if fixed_conv: name_add = 'fixed_conv_'

all = {}
for i in range(len(experiment_names)):

    with open('output/' + experiment_names[i] + '/final_accuracies_{}_{}over_training_time.pickle'.format('random', name_add), 'rb') as f:
        first = pickle.load(f)
        print(first)


    with open('output/' + experiment_names[i] + '/final_accuracies_{}over_training_time.pickle'.format(name_add), 'rb') as f:
        second = pickle.load(f)
        print(second)

    first['random init ' + plot_labels[i]] = first.pop('random')
    second['novel-activation init ' + plot_labels[i]] = second.pop('fitness')
    all.update(first)
    all.update(second)

# print(first)
cut_off_beginning = 0
helper.plot_mean_and_bootstrapped_ci_multiple(title='Accuracy of networks on CIFAR-10 training data, fixed conv layers', input_data=[np.transpose(x)[cut_off_beginning:] for k, x in all.items()], name=[k for k,x in all.items()], x_label="Epoch", y_label="Accuracy", compute_CI=True, show=True)