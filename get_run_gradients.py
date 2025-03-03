# Visualize activations and filters for evolved and random filters

import matplotlib.pyplot as plt
import argparse
import helper_hpc as helper
import wandb
from net import Net
import numpy as np
from scipy.stats import ranksums


# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--run_ids_0', nargs='+', type=str, help="enter id for first list of wandb experiments to link config")
parser.add_argument('--run_ids_1', nargs='+', type=str, help="enter id for second list of wandb experiment to link config")
# parser.add_argument('--val_acc_range', nargs=2, type=int, help='range of values to consider from array of val_acc')
# parser.add_argument('--diversity', help='run ranksums for diversity instead of accuracy', action='store_true', default=False)
args = parser.parse_args()

def run():
    
    helper.run(seed=False)

    epoch_range = [0,None]
    # if args.val_acc_range:
    #     epoch_range = args.val_acc_range

    values_0 = []
    values_1 = []
    
    counts = {}
    histogram_values = {}
    # log variables to config
    for i in range(len(args.run_ids_0)):
        run_id = args.run_ids_0[i]
        api = wandb.Api()
        run = api.run("jdonovan/novel-feature-detectors/" + run_id)
        # search = 'val_acc' if not args.diversity else 'val_novelty'
        hist = run.scan_history()
        history = [row for row in hist]
        
        # if i == 0:
        #     for column in [row for row in history][0]:
        #         if 'gradient' in column and 'weight' in column and counts.get(column) == None:
        #             counts[column] = []
        #             histogram_values[column] = []
                # if len(list(counts.keys())) > 1:
                #     break
        
        for column in [row for row in history[0]]:
            if 'gradient' in column and 'weight' in column:
                col_name = column.split('model.')[1]
                if counts != None and col_name not in counts and counts.get(col_name) == None:
                    counts[col_name] = []
                    histogram_values[col_name] = []

                grad_min = np.array([row[column]['packedBins']['min'] for row in history if row[column] != None])
                grad_size = np.array([row[column]['packedBins']['size'] for row in history if row[column] != None])

                if counts[col_name] == []:
                    histogram_values[col_name] = [grad_min[i] + (grad_size[i] * i) for i in range(len(grad_min))]
                    counts[col_name] = [row[column]['values'] for row in history if row[column] != None]
                else:
                    histogram_values[col_name] = histogram_values[col_name] + np.array([grad_min[i] + (grad_size[i] * i) for i in range(len(grad_min))])
                    counts[col_name] = counts[col_name] + np.array([row[column]['values'] for row in history if row[column] != None])
        
        # for layer in list(counts.keys()):
        #     # if gradient == "gradients/model.features.41.weight":
        #     #     print(gradient, len([(i, row[gradient]) for i, row in enumerate(history) if row[gradient] != None]))
        #     # print([row[layer] for row in history if row[layer] != None][0])
        #     print(layer)
        #     grad_min = np.array([row[layer]['packedBins']['min'] for row in history if row[layer] != None])
        #     grad_size = np.array([row[layer]['packedBins']['size'] for row in history if row[layer] != None])
        #     grad_hist_len = [row[layer]['packedBins']['count'] for row in history if row[layer] != None][0]
        #     if counts[layer] == []:
        #         histogram_values[layer] = [grad_min[i] + (grad_size[i] * i) for i in range(len(grad_min))]
        #         counts[layer] = [row[layer]['values'] for row in history if row[layer] != None]
        #     else:
        #         histogram_values[layer] = histogram_values[layer] + np.array([grad_min[i] + (grad_size[i] * i) for i in range(len(grad_min))])
        #         counts[layer] = counts[layer] + np.array([row[layer]['values'] for row in history if row[layer] != None])
        #     ref_grad = layer

    mean_gradients = {}
    for layer in counts.keys():
        for i in range(len(counts[layer])):
            mean_gradients[layer] = []
    for layer in counts.keys():
        for i in range(len(counts[layer])):
            mean_gradients[layer].append(np.mean(np.array(counts[layer][i]).astype(int)*np.array(histogram_values[layer][i])/len(args.run_ids_0)))

    std_gradients = {}
    for layer in counts.keys():
        for i in range(len(counts[layer])):
            std_gradients[layer] = []
    for layer in counts.keys():
        for i in range(len(counts[layer])):
            std_gradients[layer].append(np.mean(np.array(np.std(counts[layer][i]).astype(int)*np.array(histogram_values[layer][i]))))

    # mean_gradients = {key: [np.mean(gradients[key][index]) for index in range(len(gradient[key]))] for key in gradients.keys()}
    # std_gradients = {key: np.std(gradients[key]) for key in gradients.keys()}
        
    # print(counts)
    # print(mean_gradients)

    for layer in counts.keys():
        plt.plot(mean_gradients[layer], label=layer)
    plt.legend()
    plt.show()

    for layer in counts.keys():
        plt.plot(std_gradients[layer], label=layer)
    plt.legend()
    plt.show()
    # print(std_gradients)
    # print(len(gradients[ref_grad]))
        # print(history['gradients/model.features.41.weight'][5])
        # values = [row[search] for row in history if not np.isnan(row[search])]
        # print(len(values))
        # values_0.extend(values[epoch_range[0]:epoch_range[1]])

    # for i in range(len(args.run_ids_1)):
    #     run_id = args.run_ids_1[i]
    #     api = wandb.Api()
    #     run = api.run("jdonovan/novel-feature-detectors/" + run_id)
    #     search = 'val_acc' if not args.diversity else 'val_novelty'
    #     history = run.scan_history(keys=[search])
    #     values = [row[search] for row in history if not np.isnan(row[search])]
    #     print(len(values))
    #     values_1.extend(values[epoch_range[0]: epoch_range[1]])



    # # run ranksums test
    # print(values_1, '\n', values_0)
    # print(ranksums(values_1, values_0))
    


if __name__ == '__main__':
    run()