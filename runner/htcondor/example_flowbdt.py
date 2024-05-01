import copy
from functools import partial

import numpy as np
import torch

from joblib import Parallel, delayed
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm,trange
import time, sys, copy
import pickle

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--iter', type=int,default=0, help='iter')
flags = parser.parse_args()

# set seed
seed = 1980
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

##
X_train = np.load('your_file_{}.npy'.format(flags.iter))
y_train = np.load('your_label_{}.npy'.format(flags.iter))
mask_y = np.load('your_mask.npy')
#y=np.zeros(d*duplicate*k)
#y_uniques, y_probs = np.unique(y, return_counts=True)
#n_t = 1  separate your timestep
#b = d
#c = n  your feature dimension
#duplicate_K = 100
# XGBoost hyperparameters

max_depth = 4
n_estimators = 100
eta = 0.1
tree_method = "hist"
reg_lambda = 0.1
reg_alpha = 0.2
subsample = 1.0
def train_parallel(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        objective="reg:squarederror",
        eta=eta,
        max_depth=max_depth,
        n_jobs=16,
        #multi_strategy='one_output_tree',
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        seed=666,
        tree_method=tree_method,
        device="cpu",
    )

    #y_no_miss = ~np.isnan(y_train)
    model.fit(X_train, y_train)

    return model

# Train all model(s); fast if you have a decent multi-core CPU, but extremely slow on Google Colab because it uses a weak 2-core CPU

regr = Parallel(n_jobs=1)(  # using all cpus
    delayed(train_parallel)(
        X_train.reshape(n_t, b * duplicate_K, c)[i][mask_y, :],
        y_train.reshape(n_t, b * duplicate_K, c)[i][mask_y, k],
    )
    for i in trange(n_t)
    for k in trange(c)
)

# Replace fits with doubly loops to make things easier


regr_ = [[[None for k in range(c)] for i in range(n_t)] for j in y_uniques]
current_i = 0
for i in range(n_t):
    for j in range(len(y_uniques)):
        for k in range(c):
            regr_[j][i][k] = regr[current_i]
            current_i += 1
regr = regr_

# Save the 'regr' list
with open('models_{}.pkl'.format(flags.iter), 'wb') as f:
    pickle.dump(regr, f)
