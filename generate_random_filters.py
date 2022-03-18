import evolution as evol
from tqdm import tqdm
import helper_hpc as helper
import os
import pickle
import numpy as np
import torch
import argparse
import time

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--dataset', help='which dataset should be used for novelty metric, choices are: random, cifar-10', default='random')
parser.add_argument('--experiment_name', help='experiment name for saving and data related to filters generated', default='')
parser.add_argument('--population_size', help='number of filters to generate', type=int, default=50)
args = parser.parse_args()

def run():
    torch.multiprocessing.freeze_support()
    helper.run()
    
    global trainloader
    if args.dataset.lower() != 'cifar-10':
        random_image_paths = helper.create_random_images(64)
        trainloader = helper.load_random_images(random_image_paths)
    else:
        trainloader = helper.load_CIFAR_10()[2]
    
    global experiment_name
    
    experiment_name = args.experiment_name    
    population_size = args.population_size
    population = []

    for i in tqdm(range(population_size)): #while len(population) < population_size:
        model = evol.Model()
        model.filters = helper.get_random_filters()
        # TODO: The below funciton is far too slow
        model.activations = helper.get_activations(trainloader, model.filters)
        model.fitness = evol.compute_feature_novelty(model.activations)
        population.append(model)
        
    sols = [p.filters for p in population]
    solutions = np.array([[evol.Model() for i in range(population_size)]for j in range(1)], dtype=object)
    for i in range(1):
        solutions[i] = sols
    # solutions = solutions[0]
    sol_dict = {'random': solutions}
    fitnesses = [p.fitness for p in population]

    if not os.path.isdir('output/' + experiment_name):
        os.mkdir('output/' + experiment_name)
    with open('output/' + experiment_name + '/random_gen_solutions.pickle', 'wb') as f:
        pickle.dump(sol_dict, f)
    with open('output/' + experiment_name + '/random_gen_fitnesses.txt', 'a+') as f:
        f.write(str(fitnesses))

    fitnesses = np.array([fitnesses])
    cut_off_beginning = 0
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for x in fitnesses], name=[i for i in range(len(fitnesses))], x_label="Generation", y_label="Fitness", compute_CI=True)



if __name__ == '__main__':
    run()
