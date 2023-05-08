# import any dependencies
import numpy as np
import helper_hpc as helper
import os
from model import Model
import torch
from ae_net import AE

def run():
    torch.multiprocessing.freeze_support()
    helper.run(seed=False)
    
    # create empty holder
    solution_results= {}
    solution_results['ae_unsup'] = np.array([[Model() for i in range(1)]for j in range(5)], dtype=object)

    # loop through the models created
    for i in range(5):

        # for each grab the convolutional filter weights from them
        try:
            path = os.path.join('trained_models/trained/ae_ecifar100_ae_r{}.pth'.format(i), 'novel-feature-detectors/', os.listdir('trained_models/trained/ae_ecifar100_ae_r{}.pth/novel-feature-detectors/'.format(i))[0], 'checkpoints/epoch=63-step=40000.ckpt')
        except:
            path = os.path.join('trained_models/trained/ae_ecifar100_ae_r{}.pth'.format(i), 'novel-feature-detectors/', os.listdir('trained_models/trained/ae_ecifar100_ae_r{}.pth/novel-feature-detectors/'.format(i))[1], 'checkpoints/epoch=63-step=40000.ckpt')
        m = AE.load_from_checkpoint(path)

        with torch.no_grad():
            filters = m.get_filters(numpy=True)

            # add these to a numpy array with the same structure as the ones created by evolution
            solution_results['ae_unsup'][i] = [filters]

    # save this np array to file using npy.save (make sure the experiment name / save location is correct)
    if not os.path.isdir('output/cifar100_ae'):
        os.mkdir('output/cifar100_ae')
    with open('output/cifar100_ae/solutions_over_time_{}.npy'.format('ae_unsup'), 'wb') as f:
        np.save(f, solution_results['ae_unsup'])

if __name__ == '__main__':
    run()