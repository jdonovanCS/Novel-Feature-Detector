#!/usr/bin/env python
# coding: utf-8

import random
import time
from copy import deepcopy
import os
import numpy as np
import collections
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle
from tqdm import tqdm


# TODO: Why not use gradient descent since fitness function is differentiable. Should probably compare to that.

class Model(object):
    def __init__(self):
        self.filters = None
        self.activations = None
        self.fitness = None
        # self.novelty = None

def compute_feature_novelty(activations):
    dist = {}
    avg_dist = {}
    # for each conv layer
    for layer in activations:
        # for each activation 3d(batch, h, w)
        for batch in activations[layer]:
            # for each activation
            for ind_activation in batch:
                
                for ind_activation2 in batch:
                    if str(layer) not in dist:
                        dist[str(layer)] = []
                    dist[str(layer)].append(np.abs(ind_activation2 - ind_activation))
        avg_dist[str(layer)] = np.mean(dist[str(layer)])
    return(sum(avg_dist.values()))

def mutate(filters):
    # select a single 3x3 filter in one of the convolutional layers and replace it with a random new filter.
    selected_layer = random.randint(0,len(filters)-1)
    selected_dims = []
    for v in list(filters[selected_layer].shape)[0:2]:
        selected_dims.append(random.randint(0,v-1))
    
    selected_filter = filters[selected_layer][selected_dims[0]][selected_dims[1]]
    
    # create new random filter to replace the selected filter
    # selected_filter = torch.tensor(np.random.rand(3,3), device=helper.device)
    
    # modify the entire layer / filters by a small amount
    selected_filter += torch.rand(selected_filter.shape[0], selected_filter.shape[1])*2-1
    
    # normalize entire filter so that values are between -1 and 1
    # selected_filter = (selected_filter/np.linalg.norm(selected_filter))*2
    
    # normalize just the values that are outside of -1, 1 range
    selected_filter[(selected_filter > 1) | (selected_filter < -1)] /= torch.amax(torch.absolute(selected_filter))
    
    filters[selected_layer][selected_dims[0]][selected_dims[1]] = selected_filter
    return filters


def evolution(generations, population_size, num_children, tournament_size, num_winners=1, evolution_type="fitness"):
    """Evolutionary Algorithm

    Args:
    generations: the number of generations the algorithm should run for.
    population_size: the number of individuals to keep in the population.
    tournament_size: the number of individuals that should participate in each
        tournament.

    Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
    """
    population = collections.deque()
    solutions_over_time = []
    fitness_over_time = []

    # Initialize the population with random models.
    print("Initializing")
    for i in tqdm(range(population_size)): #while len(population) < population_size:
        model = Model()
        model.filters = helper.get_random_filters()
        model.activations = helper.get_activations(trainloader, model.filters)
        model.fitness = compute_feature_novelty(model.activations)
        population.append(model)
        
    print("Generations")
    for i in tqdm(range(generations)):
        
        parents = []  
        while len(parents) < num_children and evolution_type != "random":
        # Sample randomly chosen models from the current population.
            tournament = []
            while len(tournament) < tournament_size:
            
                candidate = random.choice(list(population))
                tournament.append(candidate)

            # The parent is the best model in the sample.
            parents.extend(sorted(tournament, key=lambda i: i.fitness, reverse=True)[:num_winners])
        

        # Create the child model and store it.
        for parent in parents:
            child = Model()
            child.filters = mutate(parent.filters)
            child.activations = helper.get_activations(trainloader, child.filters)
            child.fitness = compute_feature_novelty(child.activations)
            population.append(child)
            
        if evolution_type == 'fitness':
            population = sorted(population, key=lambda i: i.fitness, reverse=True)[:population_size]
        
        fitness_over_time.append((sorted(population, key=lambda i: i.fitness, reverse=True)[0].fitness))
        solutions_over_time.append((sorted(population, key=lambda i: i.fitness, reverse=True)[0].filters))
        
    return solutions_over_time, np.array(fitness_over_time)

def run():
    torch.multiprocessing.freeze_support()
    helper.run()
    
    random_image_paths = helper.create_random_images(64)
    global trainloader
    # trainloader = helper.load_random_images(random_image_paths)
    trainloader = helper.load_CIFAR_10()[2]
    global experiment_name
    experiment_name = "mutation_multiplier_cifar10novelty_pop20_gen50"
    # filters = helper.get_random_filters()
    # activations = helper.get_activations(trainloader, filters)

    # num_runs = 20
    num_runs = 5
    run_id = 0
    # n_iters = 100
    n_iters = 50
    output_path = './'
    # pop_size = 50
    pop_size = 20
    # tournament_size = 10
    tournament_size = 4
    # num_children = 50
    num_children = 20
    # num_winners = 5
    num_winners = 2

    fitness_results = {}
    solution_results = {}

    for run_name in ['fitness']:
        fitness_results[run_name] = np.zeros((num_runs, n_iters))
        solution_results[run_name] = np.array([[Model() for i in range(n_iters)]for j in range(num_runs)], dtype=object)
        
        print("Running Evolution for {}".format(run_name))
        
        for run_num in tqdm(range(num_runs)):
            start_time = time.time()
            solution_over_time, fitness_over_time = evolution(generations=n_iters, population_size=pop_size, num_children=num_children, tournament_size=tournament_size, num_winners=num_winners, evolution_type=run_name)
            fitness_results[run_name][run_num] = fitness_over_time
            solution_results[run_name][run_num] = solution_over_time
            res = [solution_results, fitness_results]
            print(run_name, run_num, time.time()-start_time, fitness_over_time[-1])
            with open('output.txt', 'a+') as f:
                f.write('run_name, run_num, time, fittest individual\n{}, {}, {}, {}'.format(run_name, run_num, time.time()-start_time, fitness_over_time[-1]))

    if not os.path.isdir('output/' + experiment_name):
        os.mkdir('output/' + experiment_name)
    with open('output/' + experiment_name + '/solutions_over_time.pickle', 'wb') as f:
        pickle.dump(solution_results, f)
    with open('output/' + experiment_name + '/fitness_over_time.txt', 'a+') as f:
        f.write(str(fitness_results))
    for k,v in solution_results.items():
        with open('output/' + experiment_name + '/solutions_over_time_{}.npy'.format(k), 'wb') as f:
            np.save(f, v)
    cut_off_beginning = 0
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in fitness_results.items()], name=[k for k,x in fitness_results.items()], x_label="Generation", y_label="Fitness", compute_CI=True, show=True)


if __name__ == '__main__':
    run()
