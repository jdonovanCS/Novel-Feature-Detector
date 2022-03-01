import evolution as evol
from tqdm import tqdm
import helper_hpc as helper
import os
import pickle
import numpy as np
import torch

def run():
    torch.multiprocessing.freeze_support()
    helper.run()
    random_image_paths = helper.create_random_images(64)
    global trainloader
    trainloader = helper.load_random_images(random_image_paths)
    global experiment_name
    experiment_name = "mutation_multiplier_small_edited_pop20_gen50"
    population_size = 50
    population = []

    for i in tqdm(range(population_size)): #while len(population) < population_size:
        model = evol.Model()
        model.filters = helper.get_random_filters()
        # model.filters = evol.mutate(model.filters)
        # model.activations = helper.get_activations(trainloader, model.filters)
        # model.fitness = evol.compute_feature_novelty(model.activations)
        population.append(model)
        
    sols = [p.filters for p in population]
    # sols = np.array([tuple(p.filters) for p in population], dtype=object)
    solutions = np.array([[evol.Model() for i in range(population_size)]for j in range(1)], dtype=object)
    for i in range(1):
        solutions[i] = sols
    solutions = solutions[0]
    print(type(solutions), type(solutions[0]), type(solutions[0][5]))
    print(solutions.shape, solutions[0].shape, solutions[0][5].shape)
    print(solutions.dtype, solutions[0].dtype, solutions[0][5].dtype)
    fitnesses = [p.fitness for p in population]

    if not os.path.isdir('output/' + experiment_name):
        os.mkdir('output/' + experiment_name)
    with open('output/' + experiment_name + '/random_gen_solutions.pickle', 'wb') as f:
        pickle.dump(solutions, f)
    with open('output/' + experiment_name + '/random_gen_fitnesses.txt', 'a+') as f:
        f.write(str(fitnesses))

    # cut_off_beginning = 0
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for x in fitnesses], name=[i for i in range(len(fitnesses))], x_label="Generation", y_label="Fitness", compute_CI=True)


if __name__ == '__main__':
    run()
