import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5

from calo_utils import *


"""
    for calochallenge low-level feature preprocessing, based on the code from CaloDiffusion

    input:
        - .hdf5 file for showers
    output:
        - numpy array stored in .npy

    usage:
        -f --file: Name and path of the input file to be evaluated.
        -d --dataset: '0': 1-pion(533), '1': 1-photon(368), '2': 2-electron(6480), '3': 3-electron(40500)
        -e --energies: discrete incident energy for unconditional training
        
"""

parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--file', '-f', default='', help='input file')
parser.add_argument('--dataset', '-d', type=int, default=1)
parser.add_argument('--energies', '-e', type=float, default=0.5, help='incident energies')
   
args = parser.parse_args()   



data = []
energies = []

#change your path here
for i, dataset in enumerate(['{}'.format(args.file)]):
    data_,e_,layers_ = DataLoader(
        os.path.join('./', dataset),
        [-1,368],
        emax = 4194.304,emin = 0.256,
        nevts = -1,
        max_deposit=3.1, #noise can generate more deposited energy than generated
        logE=True,
        showerMap = 'layer-logit-norm',

        nholdout = 0 if (i == len(['../../CaloDiffusion/dataset_1_photons_1.hdf5']) -1 ) else 0,
        dataset_num  = args.dataset,
        orig_shape = True,
    )
    if(i ==0): 
            data = data_
            energies = e_
            layers = layers_
    else:
        data = np.concatenate((data, data_))
        energies = np.concatenate((energies, e_))
        layers = np.concatenate((layers, layers_))
        

energies = np.reshape(energies,(-1))    

data = np.reshape(data, (len(data), -1))
layers = np.reshape(layers, (layers.shape[0], -1))


sel_data = []
sel_layer = []
sel_energy = []
for i in range(len(energies)):
    if np.abs(energies[i]-args.energies) <= 1e-2 :
        sel_data.append(data[i])
        sel_energy.append(energies[i])
        sel_layer.append(layers[i])

        
print('##########')
print('total dataset length:  {}'.format(len(sel_layer)))
print('##########')
      
np.save('sel_ds{}_{}_layer.npy'.format(args.dataset,str(int(sel_energy[0] * 1000))[:3]),sel_layer)
np.save('sel_ds{}_{}_data.npy'.format(args.dataset,str(int(sel_energy[0] * 1000))[:3]),sel_data)
      
      