import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import pickle

def run():
    torch.multiprocessing.freeze_support()
    pickled_filters = {}
    
    # get filters from pickle file
    with open("output/solutions_over_time.pickle", 'rb') as f:
        pickled_filters = pickle.load(f)
    
    helper.run()
    
    # get loader for train and test images and classes
    trainset, testset, trainloader, testloader, classes = helper.load_CIFAR_10(helper.batch_size)
    
    # create variables for holding metric
    training_record = {}
    overall_accuracy_record_no_conv = {}
    classwise_accuracy_record_no_conv = {}
    classlist = np.array(classes)
    
    # run training and evaluation and record metrics in above variables
    for name in pickled_filters.keys():
        overall_accuracy_record_no_conv[name] = np.zeros((len(pickled_filters[name]), len(pickled_filters[name][0])))
        classwise_accuracy_record_no_conv[name] = np.zeros((len(pickled_filters[name]), len(pickled_filters[name][0]), len(classlist)))
        training_record[name] = np.array([[dict for i in range(len(pickled_filters[name][0]))]for j in range(len(pickled_filters[name]))], dtype=dict)
        for filters_list in pickled_filters[name]:
            for i in range (len(filters_list)):
                run_num = np.where(pickled_filters[name] == filters_list)[0][0]
                save_path = "trained_models/no_conv_training/n{}_r{}_g{}.pth".format(name, run_num, i)
                print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, i, run_num))
                record_progress = helper.train_network_on_CIFAR_10(trainloader=trainloader, filters=filters_list[i], epochs=4, save_path=save_path, no_conv=True)
                record_accuracy_no_conv = helper.assess_accuracy(testloader=testloader, classes=classes, save_path=save_path, filters=filters_list[i])
                training_record[name][run_num][i] = record_progress
                overall_accuracy_record_no_conv[name][run_num][i] = record_accuracy_no_conv['overall']
                for c in classlist:
                    classwise_accuracy_record_no_conv[name][run_num][i][np.where(classlist==c)[0][0]] = record_accuracy_no_conv[c]
    with open('output/training_over_time.pickle', 'wb') as f:
        pickle.dump(training_record, f)
    with open('output/overall_accuracy_no_conv_over_time.pickle', 'wb') as f:
        pickle.dump(overall_accuracy_record_no_conv,f)
    with open('output/classwise_accuracy_no_conv_over_time.pickle', 'wb') as f:
        pickle.dump(classwise_accuracy_record_no_conv,f)

    cut_off_beginning = 0
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for k, x in overall_accuracy_record_no_conv.items()], name=[k for k,x in overall_accuracy_record_no_conv.items()], x_label="Generation", y_label="Fitness", compute_CI=True)

if __name__ == '__main__':
    run()
