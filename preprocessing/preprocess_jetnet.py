from __future__ import annotations  # for ArrayLike type in docs

import argparse
import os
import numpy as np
import pytorch_lightning as pl
import torch as T
import h5py
from jet_substructure import *

import awkward as ak
from coffea.nanoevents.methods import vector
from energyflow import EFPSet
from numpy.typing import ArrayLike


"""
    for jetnet high-level feature preprocessing, based on the code from JetNet and EPiC-FM

    input:
        - .hdf5 file for different jet tyoes
    output:
        - numpy array stored in .npy

    usage:
        -f --file: Name and path of the input file to be evaluated.
        -m --mode: 'all': for all type of jets
                   'one': for specific file
"""






parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--file', '-f', help='input file')
parser.add_argument('--mode', '-m', default='all', choices=['all', 'one'])
   
args = parser.parse_args()    
    
    
    
def numpy_particle(
    csts: np.ndarray,
    mask: np.ndarray,
    pt_logged=False,
) -> np.ndarray:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - del_eta
    - del_phi
    - log_pt or just pt depending on pt_logged
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = np.exp(csts[..., 2]) * mask if pt_logged else csts[..., 2]
    

    
    return np.vstack([eta, phi, pt]).T





def efps(
    jets: np.ndarray,
    use_particle_masses: bool = False,
    efpset_args: list | None = None,
    efp_jobs: int | None = None,
) -> np.ndarray:
    """
    Utility for calculating EFPs for jets in JetNet format using the energyflow library.

    Args:
        jets (np.ndarray): array of either a single or multiple jets, of shape either
          ``[num_particles, num_features]`` or ``[num_jets, num_particles, num_features]``,
          with features in order ``[eta, phi, pt, (optional) mass]``. If no particle masses given,
          they are assumed to be 0.
        efpset_args (List): Args for the energyflow.efpset function to specify which EFPs to use,
          as defined here https://energyflow.network/docs/efp/#efpset.
          Defaults to the n=4, d=5, prime EFPs.
        efp_jobs (int): number of jobs to use for energyflow's EFP batch computation.
          None means as many processes as there are CPUs.

    Returns:
        np.ndarray:
          1D (if inputted single jet) or 2D array of shape ``[num_jets, num_efps]`` of EFPs per jet

    """

    if efpset_args is None:
        efpset_args = [("n==", 4), ("d==", 4), ("p==", 1)]
    assert len(jets.shape) == 2 or len(jets.shape) == 3, "jets dimensions are incorrect"
    assert jets.shape[-1] - int(use_particle_masses) >= 3, "particle feature format is incorrect"

    efpset = EFPSet(*efpset_args, measure="hadr", beta=1, normed=None, coords="ptyphim")

    if len(jets.shape) == 2:
        # convert to energyflow format
        jets = jets[:, [2, 0, 1]] if not use_particle_masses else jets[:, [2, 0, 1, 3]]
        efps = efpset.compute(jets)
    else:
        # convert to energyflow format
        jets = jets[:, :, [2, 0, 1]] if not use_particle_masses else jets[:, :, [2, 0, 1, 3]]
        efps = efpset.batch_compute(jets, efp_jobs)

    return efps
def locals_to_rel_mass_and_efp(csts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert the values of a set of constituents to the relative mass and EFP
    values of the jet they belong to.

    Args:
        csts: A numpy array of shape (batch_size, n_csts, 3)
            containing the (eta, phi, pt) values of the constituents.
        mask: A numpy array of shape (batch_size, n_csts)
            containing a mask for the constituents, used to sum only over
            the valid constituents.

    Returns:
        A numpy array of shape (batch_size, 2)
            containing the relative mass and EFP values of the jet.
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_m = np.sqrt(
        np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None)
    )

    # Get the efp values
    jet_efps = efps(csts, efp_jobs=1).mean(axis=-1)

    return np.vstack([jet_m, jet_efps]).T

if args.mode in ['all']:

    file_path = ['g.hdf5','t.hdf5','q.hdf5','w.hdf5','z.hdf5']
    types = ['g','t','q','w','z']
    counts = 0 
    for i in file_path:

        with h5py.File(i, 'r') as file:
            # Print the keys (group/dataset names) present in the HDF5 file
            print("Keys: %s" % list(file.keys()) + "for {}".format(i))

            # Access a dataset (replace 'dataset_name' with the actual dataset name)
            dataset = file['particle_features']
            # Read data from the dataset
            data = dataset[()]  # This reads the entire dataset into a NumPy array
            data = data.astype(np.float32)
            h5_file = "generated/{}_substructure.h5".format(types[counts])
            h5file_path = Path(h5_file)

            print('Start relative substructure')


            dump_hlvs(data, h5file_path, plot=True)

            mask = ~np.all(data == 0, axis=-1)

            print('Start relative mass and efp')
            mefp = locals_to_rel_mass_and_efp(data, mask)



        with h5py.File('{}_mefp.h5'.format(types[counts]), mode="w") as output_file:
            output_file.create_dataset('relative_m', data=mefp[:, 0])
            output_file.create_dataset('efp', data=mefp[:, 1])



        counts += 1

if args.mode in ['one']:
    file_path = ['{}'.format(args.file)]
    for i in file_path:

        with h5py.File(i, 'r') as file:
            # Print the keys (group/dataset names) present in the HDF5 file
            print("Keys: %s" % list(file.keys()) + "for {}".format(i))

            # Access a dataset (replace 'dataset_name' with the actual dataset name)
            dataset = file['particle_features']
            # Read data from the dataset
            data = dataset[()]  # This reads the entire dataset into a NumPy array
            data = data.astype(np.float32)
            h5_file = "generated/substructure.h5"
            h5file_path = Path(h5_file)

            print('Start relative substructure')


            dump_hlvs(data, h5file_path, plot=True)

            mask = ~np.all(data == 0, axis=-1)

            print('Start relative mass and efp')
            mefp = locals_to_rel_mass_and_efp(data, mask)



        with h5py.File('mefp.h5', mode="w") as output_file:
            output_file.create_dataset('relative_m', data=mefp[:, 0])
            output_file.create_dataset('efp', data=mefp[:, 1])

