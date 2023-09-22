import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import argparse
import random
from functools import partial

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--rand_tech', help="random technique used to generate filters", type=str, default=None)
parser.add_argument('--gram-schmidt', help='gram-schmidt used to orthonormalize filters', action='store_true')
parser.add_argument('--unique_id', help='if a unique id is associated with the file the solution is stored in give it here.', default="", type=str)
parser.add_argument('--no-evo', action='store_true')
args = parser.parse_args()

def run():

    print('running')
    torch.multiprocessing.freeze_support()

    stored_filters = {}
    
    experiment_name = args.experiment_name
    
    name = 'fitness'
    if args.gram_schmidt:
        name = 'gram-schmidt'
    if args.rand_tech:
        name=args.rand_tech
    if args.unique_id != "":
        name = 'current_' + name + "_" + args.unique_id
    
    filename = ''
    filename='output/' + experiment_name + '/solutions_over_time_{}.npy'.format(name)
    random_filename='output/' + experiment_name + '/solutions_over_time_{}.npy'.format(name)

    # get filters from numpy file
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    stored_filters = np.load(filename)
    stored_filters_random = np.load(random_filename)
    np.load = np_load_old

    if args.unique_id != '':
        stored_filters = [stored_filters]

    
    print(stored_filters_random.shape)
    random.shuffle(stored_filters_random[0])
    stored_filters_random = np.array(stored_filters_random)
    with open(filename, 'wb') as f:
        np.save(f, stored_filters)
    
    helper.run(seed=False)

    import matplotlib.pyplot as plt
    from scipy.interpolate import UnivariateSpline
    from scipy.stats.kde import gaussian_kde
    num_runs = len(stored_filters)
    pdf = pdf_r = 0
    for layer in range(len(stored_filters[0][0])):
        pdf = pdf_r = mean = mean_r = std = std_r = 0
        divisor = max(abs(stored_filters_random[0][0][layer].flatten()))
        multiplier = max(abs(stored_filters[0][0][layer].flatten()))
        num_bins_ = int(100*multiplier/divisor)
        num_outside = 0
        for run_num in range(num_runs):
            filters = stored_filters[run_num][0]
            filters_random = stored_filters_random[run_num][0]
            
            num_outside += sum(abs(filters[layer].flatten()) > divisor)

            # max(np.abs(filters[layer].flatten()))
            # getting data of the histogram
            count_r, bins_count_r = np.histogram(filters_random[layer].flatten(), bins=100, normed=True)
            count, bins_count = np.histogram(filters[layer].flatten(), bins=num_bins_, normed=True)
            
            # verify sum to 1
            widths = bins_count[1:] - bins_count[:-1]
            widths_r = bins_count_r[1:]-bins_count_r[:-1]
            assert sum(count * widths) > .99 and sum(count * widths) < 1.01
            assert sum(count_r * widths_r) > .99 and sum(count_r * widths_r) < 1.01


            # finding the PDF of the histogram using count values
            pdf += count / sum(count)
            pdf_r += count_r/sum(count_r)

            mean += filters[layer].flatten().mean()
            mean_r += filters_random[layer].flatten().mean()
            std += filters[layer].flatten().std()
            std_r += filters_random[layer].flatten().std()
            
            # using numpy np.cumsum to calculate the CDF
            # We can also find using the PDF values by looping and adding
        cdf = np.cumsum(pdf)
        cdf_r = np.cumsum(pdf_r)
            
        pdf = pdf / len(stored_filters[0][0])
        pdf_r = pdf_r / len(stored_filters[0][0])
        mean = mean / len(stored_filters[0][0])
        mean_r = mean_r / len(stored_filters[0][0])
        std = std / len(stored_filters[0][0])
        std_r = std_r / len(stored_filters[0][0])
        # maximum = [max(stored_filters[0][0][l]) for l in range(len(stored_filters[0][0]))]
        # minimum = [min(stored_filters[0][0][l]) for l in range(len(stored_filters[0][0]))]
        # plotting PDF and CDF
        # Attempt at smoothing
        # bins_count = bins_count[:-1] + (bins_count[1] - bins_count[0])/2   # convert bin edges to centers
        # f = UnivariateSpline(bins_count, count, s=100)
        # plt.plot(bins_count, f(bins_count), color="blue", label="PDF_{}".format(layer))
        # bins_count_r = bins_count_r[:-1] + (bins_count_r[1] - bins_count_r[0])/2   # convert bin edges to centers
        # f = UnivariateSpline(bins_count_r, count_r, s=100)
        # plt.plot(bins_count_r, f(bins_count_r), color="red", label="PDF_r_{}".format(layer))

        # Another attempt
        # kde = gaussian_kde( filters[layer].flatten() )
        # kde_r = gaussian_kde(filters_random[layer].flatten())
        # # these are the values over wich your kernel will be evaluated
        # dist_space = np.linspace( min(filters[layer].flatten()), max(filters[layer].flatten()), 100 )
        # dist_space_r = np.linspace( min(filters_random[layer].flatten()), max(filters_random[layer].flatten()), 100 )
        # plt.plot( dist_space, kde(dist_space) )
        # plt.plot( dist_space_r, kde_r(dist_space_r) )
        
        # Original plots
        if not args.no_evo:
            plt.plot(bins_count[1:], pdf, color="blue", label="PDF_{}".format(layer+1))
        plt.plot(bins_count_r[1:], pdf_r, color="red", label="PDF_r_{}".format(layer+1))
        # plt.plot(bins_count[1:], cdf, label="CDF")
        plt.legend()
        # print(mag)
        perc_outside = num_outside/len(stored_filters[0][0][layer].flatten())/(num_runs)
        perc_inside = 1-perc_outside
        # print('percentage of weights outside of the range: -{}, {}: {}'.format(divisor, divisor, perc_outside))
        # print('percentage of weights inside of the range: -{}, {}: {}'.format(divisor, divisor, perc_inside))
        print('mean: {} mean_random: {} \t std: {} std_random: {}'.format(mean, mean_r, std, std_r))
        plt.show()

if __name__ == '__main__':
    run()