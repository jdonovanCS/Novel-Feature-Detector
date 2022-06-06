#!/usr/bin/env python
# coding: utf-8

import random
import time
import os
import numpy as np
import collections
import warnings

from pytorch_lightning import Trainer
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle
from tqdm import tqdm
import argparse
import gc
from model import Model
import numba
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl


# TODO: Why not use gradient descent since fitness function is differentiable. Should probably compare to that.


parser=argparse.ArgumentParser(description="Process some inputs")
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--batch_size', help="batch size for computing novelty, only 1 batch is used", type=int, default=64)
parser.add_argument('--evo_gens', type=int, help="number of generations used in evolving solutions", default=None)
parser.add_argument('--evo_pop_size', type=int, help='Number of individuals in population when evolving solutions', default=None)
parser.add_argument('--evo_dataset_for_novelty', help='Dataset used for novelty computation during evolution and training', default=None)
parser.add_argument('--evo_num_runs', type=int, help='Number of runs used in evolution', default=None)
parser.add_argument('--evo_tourney_size', type=int, help='Size of tournaments in evolutionary algorithm selection', default=None)
parser.add_argument('--evo_num_winners', type=int, help='Number of winners in tournament in evolutionary algorithm', default=None)
parser.add_argument('--evo_num_children', type=int, help='Number of children in evolutionary algorithm', default=None)
parser.add_argument('--diversity_type', default='absolute', type=str, help='Type of diversity metric to use for this experiment (ie. relative, absolute, original, etc.)')    
args = parser.parse_args()

# @numba.njit(parallel=True)
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
    # data_iterator = iter(data_module.train_dataloader())
    # net_input = next(data_iterator)

    print("\nInitializing")
    for i in tqdm(range(population_size)): #while len(population) < population_size:
        model = Model()
        net = helper.Net(num_classes=len(classnames), classnames=classnames, diversity=args.diversity_type)
        model.filters = net.get_filters()
        trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
        model.fitness =  net.avg_novelty
        population.append(model)
        helper.wandb.log({'gen': 0, 'individual': i, 'fitness': model.fitness})
        
    print("\nGenerations")
    for i in tqdm(range(generations)):
        # net_input = next(data_iterator)
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
            net.set_filters(child.filters)
            trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
            child.fitness = net.avg_novelty
            population.append(child)
            
        if evolution_type == 'fitness':
            population = sorted(population, key=lambda i: i.fitness, reverse=True)[:population_size]
            for j in range(len(population)):
                helper.wandb.log({'gen': i, 'individual': j, 'fitness': population[j].fitness})
        
        best_fitness = sorted(population, key=lambda i: i.fitness, reverse=True)[0].fitness
        best_solution = sorted(population, key=lambda i: i.fitness, reverse=True)[0].filters
        fitness_over_time.append((best_fitness))
        solutions_over_time.append((best_solution))
        helper.wandb.log({'gen': i, 'best_individual_fitness': best_fitness})
        # helper.wandb.log({'gen': i, 'best_individual_filters': best_solution})
        
    return solutions_over_time, np.array(fitness_over_time)

def run():
    torch.multiprocessing.freeze_support()
    helper.run(seed=False)
    
    helper.config['batch_size'] = args.batch_size
    helper.config['experiment_name'] = args.experiment_name
    helper.config['evo_gens'] = args.evo_gens
    helper.config['evo_pop'] = args.evo_pop_size
    helper.config['evo_dataset_for_novelty'] = args.evo_dataset_for_novelty
    helper.config['evo_num_runs'] = args.evo_num_runs
    helper.config['evo_tourney_size'] = args.evo_tourney_size
    helper.config['evo_num_winners'] = args.evo_num_winners
    helper.config['evo_num_children'] = args.evo_num_children
    helper.config['diversity_type'] = args.diversity_type
    helper.config['experiment_type'] = 'evolution'
    helper.update_config()

    random_image_paths = helper.create_random_images(64)
    global data_module
    data_module = helper.get_data_module(args.evo_dataset_for_novelty, batch_size=args.batch_size)
    data_module.prepare_data()
    data_module.setup()
    # data_iterator = iter(data_module.train_dataloader())
    # global net_input
    # net_input = next(data_iterator)
    # wandb_logger = WandbLogger(log_model=True)
    global trainer
    # trainer = pl.Trainer(logger=wandb_logger, accelerator="auto")
    trainer = pl.Trainer(accelerator="auto")
    global classnames
    classnames = list(data_module.dataset_test.classes)

    # global trainloader
    # trainloader = helper.load_random_images(random_image_paths)
    # trainloader = helper.load_CIFAR_10()[2]
    global experiment_name
    experiment_name = args.experiment_name

    # num_runs = 20
    num_runs = args.evo_num_runs
    run_id = 0
    # n_iters = 100
    n_iters = args.evo_gens
    output_path = './'
    # pop_size = 50
    pop_size = args.evo_pop_size
    # tournament_size = 10
    tournament_size = args.evo_tourney_size
    # num_children = 50
    num_children = args.evo_num_children
    # num_winners = 5
    num_winners = args.evo_num_winners

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
            helper.run(seed=False)
            helper.config['batch_size'] = args.batch_size
            helper.config['experiment_name'] = args.experiment_name
            helper.config['evo_gens'] = args.evo_gens
            helper.config['evo_pop'] = args.evo_pop_size
            helper.config['evo_dataset_for_novelty'] = args.evo_dataset_for_novelty
            helper.config['evo_num_runs'] = args.evo_num_runs
            helper.config['evo_tourney_size'] = args.evo_tourney_size
            helper.config['evo_num_winners'] = args.evo_num_winners
            helper.config['evo_num_children'] = args.evo_num_children
            helper.config['diversity_type'] = args.diversity_type
            helper.config['experiment_type'] = 'evolution'
            helper.update_config()
            
    if not os.path.isdir('plots/' + experiment_name):
        os.mkdir('plots/' + experiment_name)
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
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in fitness_results.items()], name=[k for k,x in fitness_results.items()], x_label="Generation", y_label="Fitness", compute_CI=True, save_name=experiment_name + "/fitness_over_time.png")
    # os.system('conda activate EC2')
    # # os.system('python evolution.py --experiment_name=\"further testing of logging with wandb and lightning\" --batch_size=64 --evo_gens=30 --evo_pop_size=10 --evo_dataset_for_novelty=cifar10 --evo_num_runs=1 --evo_tourney_size=4 --evo_num_winners=2 --evo_num_children=10')
    # os.system('python train_and_eval.py --dataset={} --experiment_name="{}" --fixed_conv --training_interval=1 --epochs=96 --novelty_interval=1 --test_accuracy_interval=4 --batch_size={} --evo_gens={} --evo_pop_size={} --evo_dataset_for_novelty={} --evo_num_runs={} --evo_tourney_size={} --evo_num_winners={} --evo_num_children={}'.format(args.evo_dataset_for_novelty, args.experiment_name, args.batch_size, args.evo_gens, args.evo_pop_size, args.evo_dataset_for_novelty, args.evo_num_runs, args.evo_tourney_size, args.evo_num_winners, args.evo_num_children))
    # os.system('python generate_random_filters.py --dataset={} --experiment_name="{}" --population_size={} --batch_size={}'.format(args.evo_dataset_for_novelty, experiment_name, 50, args.batch_size))
    # os.system('python train_and_eval.py --dataset={} --experiment_name="{}" --fixed_conv --training_interval=.2 --epochs=96 --novelty_interval=1 --test_accuracy_interval=4 --batch_size={} --evo_gens={} --evo_pop_size={} --evo_dataset_for_novelty={} --evo_num_runs={} --evo_tourney_size={} --evo_num_winners={} --evo_num_children={} --random'.format(args.evo_dataset_for_novelty, args.experiment_name, args.batch_size, args.evo_gens, args.evo_pop_size, args.evo_dataset_for_novelty, args.evo_num_runs, args.evo_tourney_size, args.evo_num_winners, args.evo_num_children))
    # os.system('python train_and_eval.py --dataset={} --experiment_name="{}" --training_interval=1 --epochs=32 --novelty_interval=1 --test_accuracy_interval=4 --batch_size={} --evo_gens={} --evo_pop_size={} --evo_dataset_for_novelty={} --evo_num_runs={} --evo_tourney_size={} --evo_num_winners={} --evo_num_children={}'.format(args.evo_dataset_for_novelty, args.experiment_name, args.batch_size, args.evo_gens, args.evo_pop_size, args.evo_dataset_for_novelty, args.evo_num_runs, args.evo_tourney_size, args.evo_num_winners, args.evo_num_children))
    # os.system('python train_and_eval.py --dataset={} --experiment_name="{}" --training_interval=.2 --epochs=32 --novelty_interval=1 --test_accuracy_interval=4 --batch_size={} --evo_gens={} --evo_pop_size={} --evo_dataset_for_novelty={} --evo_num_runs={} --evo_tourney_size={} --evo_num_winners={} --evo_num_children={} --random'.format(args.evo_dataset_for_novelty, args.experiment_name, args.batch_size, args.evo_gens, args.evo_pop_size, args.evo_dataset_for_novelty, args.evo_num_runs, args.evo_tourney_size, args.evo_num_winners, args.evo_num_children))
    # os.system('python train_and_eval.py --dataset=cifar100 --experiment_name="{}" --fixed_conv --training_interval=1 --epochs=1024 --novelty_interval=1 --test_accuracy_interval=4 --batch_size={} --evo_gens={} --evo_pop_size={} --evo_dataset_for_novelty={} --evo_num_runs={} --evo_tourney_size={} --evo_num_winners={} --evo_num_children={}'.format(args.evo_dataset_for_novelty, args.experiment_name, args.batch_size, args.evo_gens, args.evo_pop_size, args.evo_dataset_for_novelty, args.evo_num_runs, args.evo_tourney_size, args.evo_num_winners, args.evo_num_children))
    # os.system('python train_and_eval.py --dataset=cifar100 --experiment_name="{}" --fixed_conv --training_interval=.2 --epochs=1024 --novelty_interval=1 --test_accuracy_interval=4 --batch_size={} --evo_gens={} --evo_pop_size={} --evo_dataset_for_novelty={} --evo_num_runs={} --evo_tourney_size={} --evo_num_winners={} --evo_num_children={} --random'.format(args.evo_dataset_for_novelty, args.experiment_name, args.batch_size, args.evo_gens, args.evo_pop_size, args.evo_dataset_for_novelty, args.evo_num_runs, args.evo_tourney_size, args.evo_num_winners, args.evo_num_children))


if __name__ == '__main__':
    with torch.no_grad():
        run()
