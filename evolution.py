#!/usr/bin/env python
# coding: utf-8

import random
import time
from copy import deepcopy
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle

class Model(object):
    def __init__(self):
        self.filters = None
        self.activations = None
        self.fitness = None
        # self.novelty = None

def compute_feature_novelty(activations):
    dist = {0: [], 1: [], 2: []}
    avg_dist = {0: None, 1: None, 2: None}
    for i in range(len(activations)):
        for a in activations[i]:
            for a2 in activations[i]:
                dist[i].append(np.abs(a2 - a))
        avg_dist[i] = np.mean(dist[i])
    return(sum(avg_dist.values()))

def mutate(filters):
    # select a single 3x3 filter in one of the convolutional layers and replace it with a random new filter.
    selected_layer = random.randint(0,len(filters)-1)
    selected_dims = []
    for v in list(filters[selected_layer].shape)[0:2]:
        selected_dims.append(random.randint(0,v-1))
    print('selected_layer', selected_layer, 'selected_dims', selected_dims)
    filters[selected_layer][selected_dims[0]][selected_dims[1]] = torch.tensor(np.random.uniform(-1, 1, (3,3)), device=helper.device)
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
    while len(population) < population_size:
        model = Model()
        model.filters = helper.get_random_filters()
        model.activations = helper.get_activations(trainloader, model.filters)
        model.fitness = compute_feature_novelty(model.activations)
        population.append(model)
        
    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    for i in range(generations):
        
        parents = []  
        while len(parents) < num_children and evolution_type != "random":
        # Sample randomly chosen models from the current population.
            tournament = []
            while len(tournament) < tournament_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
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
        
        if evolution_type == 'random':
            population = [sorted(population, key=lambda i: i.fitness, reverse=True)[0]]
            while len(population) < population_size:
                model = Model()
                model.filters = helper.get_random_filters()
                model.activations = helper.get_activations(trainloader, model.filters)
                model.fitness = compute_feature_novelty(model.activations)
                population.append(model)
        
        fitness_over_time.append(1+(sorted(population, key=lambda i: i.fitness, reverse=True)[0].fitness))
        solutions_over_time.append((sorted(population, key=lambda i: i.fitness, reverse=True)[0].filters))
        
    return solutions_over_time, np.array(fitness_over_time)

def plot_mean_and_bootstrapped_ci_multiple(input_data = None, title = 'overall', name = "change this", x_label = "x", y_label = "y", save_name="", compute_CI=True, maximum_possible=None):
    """ 
     
    parameters:  
    input_data: (numpy array of numpy arrays of shape (max_k, num_repitions)) solution met
    name: numpy array of string names for legend 
    x_label: (string) x axis label 
    y_label: (string) y axis label 
     
    returns: 
    None 
    """ 
 
    generations = len(input_data[0])
 
    fig, ax = plt.subplots() 
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label) 
    ax.set_title(title) 
    for i in range(len(input_data)): 
        CIs = [] 
        mean_values = [] 
        for j in range(generations): 
            mean_values.append(np.mean(input_data[i][j])) 
            if compute_CI:
                CIs.append(bootstrap.ci(input_data[i][j], statfunction=np.mean)) 
        mean_values=np.array(mean_values) 
 
        high = [] 
        low = [] 
        if compute_CI:
            for j in range(len(CIs)): 
                low.append(CIs[j][0]) 
                high.append(CIs[j][1]) 
 
        low = np.array(low) 
        high = np.array(high) 

        y = range(0, generations) 
        ax.plot(y, mean_values, label=name[i])
        if compute_CI:
            ax.fill_between(y, high, low, alpha=.2) 
        ax.legend()
    
    if maximum_possible:
        ax.hlines(y=maximum_possible, xmin=0, xmax=generations, linewidth=2, color='r', linestyle='--', label='best poss. acc.')
        ax.legend()

    plt.savefig('fitness_over_time_plot_with_CIs.png')
    plt.show()
    

def run():
    torch.multiprocessing.freeze_support()
    helper.run()
    random_image_paths = helper.create_random_images()
    global trainloader
    trainloader = helper.load_random_images(random_image_paths)
    # filters = helper.get_random_filters()
    # activations = helper.get_activations(trainloader, filters)
    # for a in activations:
    #     print(a.shape)

    # num_runs = 20
    num_runs = 1
    run_id = 0
    # n_iters = 100
    n_iters = 10
    output_path = './'
    # pop_size = 50
    pop_size = 10
    # tournament_size = 10
    tournament_size = 2
    # num_children = 50
    num_children = 10
    # num_winners = 5
    num_winners = 1

    fitness_results = {}
    solution_results = {}

    for run_name in ['fitness', 'random']:
        fitness_results[run_name] = np.zeros((num_runs, n_iters))
        solution_results[run_name] = np.array([[Model() for i in range(n_iters)]for j in range(num_runs)], dtype=object)
        # new_output_path = os.path.join(output_path, run_name + "evolution")
        # os.makedirs(os.path.join(new_output_path), exist_ok=True)
        for run_num in range(num_runs):
            start_time = time.time()
            solution_over_time, fitness_over_time = evolution(generations=n_iters, population_size=pop_size, num_children=num_children, tournament_size=tournament_size, num_winners=num_winners, evolution_type=run_name)
            fitness_results[run_name][run_num] = fitness_over_time
            solution_results[run_name][run_num] = solution_over_time
            res = [solution_results, fitness_results]
            print(run_name, run_num, time.time()-start_time, fitness_over_time[-1])
            with open('output.txt', 'a+') as f:
                f.write('run_name, run_num, time, fittest individual\n{}, {}, {}, {}'.format(run_name, run_num, time.time()-start_time, fitness_over_time[-1]))

    with open('solutions_over_time.pickle', 'wb') as f:
        pickle.dump(solution_results, f)
    with open('fitness_over_time.txt', 'a+') as f:
        f.write(str(fitness_results))
    for k,v in solution_results.items():
        with open('solutions_over_time_{}.npy'.format(k), 'wb') as f:
            np.save(f, v)
    cut_off_beginning = 0
    plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in fitness_results.items()], name=[k for k,x in fitness_results.items()], x_label="Generation", y_label="Fitness", compute_CI=True)


if __name__ == '__main__':
    run()
