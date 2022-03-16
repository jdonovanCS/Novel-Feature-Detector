import helper_hpc as helper
import pickle
import numpy as np

experiment_name = "mutation_multiplier_cifar10novelty_pop20_gen50"
name = 'random'
no_conv = True
name_add = ''
if no_conv: name_add = 'no_conv_'
with open('output/' + experiment_name + '/final_accuracies_{}_{}over_training_time.pickle'.format(name, name_add), 'rb') as f:
    first = pickle.load(f)

name = 'fitness'
with open('output/' + experiment_name + '/final_accuracies_{}over_training_time.pickle'.format(name_add), 'rb') as f:
    second = pickle.load(f)

first.update(second)
first['random initialization'] = first.pop('random')
first['novel-activation initialization'] = first.pop('fitness')
cut_off_beginning = 0
helper.plot_mean_and_bootstrapped_ci_multiple(title='Accuracy of networks on CIFAR-10 training data using CIFAR-10 for novelty, fixed conv layers', input_data=[np.transpose(x)[cut_off_beginning:] for k, x in first.items()], name=[k for k,x in first.items()], x_label="Epoch", y_label="Accuracy", compute_CI=True, show=True)
