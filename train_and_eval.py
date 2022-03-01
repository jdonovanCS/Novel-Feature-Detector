import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle

def save_final_accuracy_of_trained_models(pickle_path, save_path):

    # read in the saved training record
    with open(pickle_path, 'rb') as f:
        pickled_metrics = pickle.load(f)

    modified_accuracies = {}
    for key in pickled_metrics:
        modified_accuracies[key] = np.zeros((len(pickled_metrics[key]), len(pickled_metrics[key][0][-1]['running_acc'])))
        for i in range(len(pickled_metrics[key])):
            modified_accuracies[key][i] = [p['accuracy'] for p in pickled_metrics[key][i][-1]['running_acc']]
    
    with open(save_path, 'wb') as f:
        pickle.dump(modified_accuracies, f)
    
    print(modified_accuracies)
    return modified_accuracies

def run():
    torch.multiprocessing.freeze_support()
    pickled_filters = {}
    experiment_name = "mutation_multiplier_small_edited_pop20_gen50"
    just_train_using_final_generation_filters = True
    no_conv = True
    
    # get filters from pickle file
    with open("output/" + experiment_name + "/solutions_over_time.pickle", 'rb') as f:
        pickled_filters = pickle.load(f)
    
    helper.run()
    
    # get loader for train and test images and classes
    trainset, testset, trainloader, testloader, classes = helper.load_CIFAR_10(helper.batch_size)
    
    # create variables for holding metric
    training_record = {}
    overall_accuracy_record = {}
    classwise_accuracy_record = {}
    classlist = np.array(classes)
    epochs = 96
    
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran
    for name in pickled_filters.keys():
        # instatiate entry in dictionary for this type of evolution
        overall_accuracy_record[name] = np.zeros((len(pickled_filters[name]), len(pickled_filters[name][0])))
        classwise_accuracy_record[name] = np.zeros((len(pickled_filters[name]), len(pickled_filters[name][0]), len(classlist)))
        training_record[name] = np.array([[dict for i in range(len(pickled_filters[name][0]))]for j in range(len(pickled_filters[name]))], dtype=dict)
        # for each run of this evolution type
        for filters_list in pickled_filters[name]:
            run_num = np.where(pickled_filters[name] == filters_list)[0][0]
            # for each generation train the solution output at that generation
            for i in range (len(filters_list)):
                # if we only want to train the solution from the final generation
                # put zeros in the metric dictionaries if this isn't the final generation
                if just_train_using_final_generation_filters and i != len(filters_list)-1:
                    overall_accuracy_record[name][run_num][i] = 0
                    for c in classlist:
                        classwise_accuracy_record[name][run_num][i][np.where(classlist==c)[0][0]] = 0
                    training_record[name][run_num][i] = {'running_acc': [], 'running_loss': []}
                    continue
                
                # else train the network and collect the metrics
                save_path = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not no_conv, experiment_name, name, run_num, i)
                print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, i, run_num))
                record_progress = helper.train_network_on_CIFAR_10(trainloader=trainloader, filters=filters_list[i], epochs=epochs, testloader=testloader, classes=classes, save_path=save_path, no_conv=no_conv)
                record_accuracy = helper.assess_accuracy(testloader=testloader, classes=classes, save_path=save_path)
                training_record[name][run_num][i] = record_progress
                overall_accuracy_record[name][run_num][i] = record_accuracy['overall']
                for c in classlist:
                    classwise_accuracy_record[name][run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]
    name_add = ''
    if no_conv: name_add = 'no_conv_'
    with open('output/' + experiment_name + '/training_{}over_time.pickle'.format(name_add), 'wb') as f:
        pickle.dump(training_record, f)
    with open('output/' + experiment_name + '/overall_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
        pickle.dump(overall_accuracy_record,f)
    with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
        pickle.dump(classwise_accuracy_record,f)

    cut_off_beginning = 0
    final_accuracies = save_final_accuracy_of_trained_models('output/' + experiment_name + '/training_{}over_time.pickle'.format(name_add), 'output/' + experiment_name + '/final_accuracies_{}over_training_time.pickle'.format(name_add))
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in overall_accuracy_record.items()], name=[k for k,x in overall_accuracy_record.items()], x_label="Generation", y_label="Fitness", compute_CI=True)
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in final_accuracies.items()], name=[k for k,x in final_accuracies.items()], x_label="Epoch", y_label="Accuracy", compute_CI=True)



if __name__ == '__main__':
    run()
