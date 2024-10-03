from tqdm import tqdm
import helper_hpc as helper
import os
import pickle
import numpy as np
import torch
import argparse
from model import Model
import copy


parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--experiment_name', help='experiment name for saving and data related to filters generated', default='')
parser.add_argument('--population_size', help='number of filters to generate', type=int, default=50)
parser.add_argument('--technique', help='uniform, normal, gram-schmidt, or mutate-only technique', type=str, default='uniform')
parser.add_argument('--network', help="specify network to generate filters for (vgg16, conv6, etc.)", type=str, default='conv6')
parser.add_argument('--mr', help="mutation rate to use if using mutation-only", default=1.0, type=float)
parser.add_argument('--broad_mut', help="use broad muation", default=False, action="store_true")
parser.add_argument('--num_mutations', help='how many times to run the mutation function on the filters', default=500, type=int)
parser.add_argument('--gain', help='optional scaling factor for some of the technqiues', default=1.0, type=float)
parser.add_argument('--weighted_mut', help="would we like for the mutation function to weight its selection based on number of filters in each layer", default=False, action='store_true')
parser.add_argument('--weights_for_mut', help='specify weights for each layer during mutation', nargs=6, default=None, type=float)
# parser.add_argument('--batch_size', help='Number of images to use for novelty metric, only 1 batch used', default=64, type=int)
# parser.add_argument('--dataset', help='which dataset should be used for novelty metric, choices are: random, cifar-10', default='random')
args = parser.parse_args()

if args.weights_for_mut and (sum(args.weights_for_mut) < 0.99999999 or sum(args.weights_for_mut) > 1.000000001):
    print('weights must add to 1.0. Current sum is: ', sum(args.weights_for_mut))
    exit()

def run():
    torch.multiprocessing.freeze_support()
    helper.run(seed=False)
    
    # global trainloader
    # if args.dataset.lower() != 'cifar-10' and args.dataset.lower() != 'cifar10':
    #     random_image_paths = helper.create_random_images(64)
    #     data_module = helper.load_random_images(random_image_paths)
    # else:
    #     data_module = helper.get_data_module(args.dataset.lower(), batch_size=args.batch_size)

    # data_module.prepare_data(data_dir="data/")
    # data_module.setup()
    # classnames = list(data_module.dataset_test.classes)

    # data_iterator = iter(data_module.train_dataloader())
    # net_input = next(data_iterator)
    
    experiment_name = args.experiment_name    
    population_size = args.population_size
    population = []
    helper.config['experiment_name'] = experiment_name

    for i in tqdm(range(population_size)): #while len(population) < population_size:
        model = Model()
        if args.network == 'vgg16':
            net = helper.BigNet()
        elif args.network == 'conv6':
            net = helper.Net()
        # model.fitness =  net.get_fitness(net_input)
        if args.technique == 'gram-schmidt':
            model.filters = net.get_filters(numpy=True)
            new_filters = helper.gram_shmidt_orthonormalize(model.filters)
            net.set_filters = new_filters
        elif  'xavier-normal' in args.technique:
            helper.xavier_normal(net, args.gain)
        elif 'xavier' in args.technique:
            helper.xavier_uniform(net, args.gain)
        elif 'orthogonal' in args.technique:
            helper.orthogonal(net, args.gain)
        elif 'default_normal' in args.technique:
            helper.default_normal(net)
        elif 'normal' in args.technique:
            helper.normalize(net)
        elif 'default uniform' in args.technique:
            helper.default_uniform(net)
        if 'mutate-only' in args.technique:
            model.filters = copy.deepcopy(net.get_filters())
            for k in range(args.num_mutations):
                model.filters = helper.mutate(model.filters, args.broad_mut, args.mr, args.weighted_mut, args.weights_for_mut)
            net.set_filters(copy.deepcopy(model.filters))
        model.filters = net.get_filters()
        population.append(model)
        # helper.wandb.log({'gen': 0, 'individual': i, 'fitness': model.fitness})
        
    # sols = [p.filters for p in population]
    sols = [p.filters for p in population]
    solutions = np.array([[Model() for i in range(1)]for j in range(population_size)], dtype=object)
    for i in range(population_size):
        solutions[i][0] = sols[i]
    # solutions = solutions[0]
    sol_dict = {args.technique: solutions}
    # fitnesses = [p.fitness for p in population]

    if not os.path.isdir('output/' + experiment_name):
        os.mkdir('output/' + experiment_name)
    # with open('output/' + experiment_name + '/random_gen_solutions.pickle', 'wb') as f:
    #     pickle.dump(sol_dict, f)
    for k,v in sol_dict.items():
        with open('output/' + experiment_name + '/solutions_over_time_{}.npy'.format(k), 'wb') as f:
            np.save(f, v[::])
    # with open('output/' + experiment_name + '/random_gen_fitnesses.txt', 'a+') as f:
    #     f.write(str(fitnesses))

    # fitnesses = np.array([fitnesses])
    # cut_off_beginning = 0
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for x in fitnesses], name=[i for i in range(len(fitnesses))], x_label="Generation", y_label="Fitness", compute_CI=True)



if __name__ == '__main__':

    run()

