import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import sys
import joblib
from sklearn.preprocessing import QuantileTransformer
sys.path.append("..")
from XMLHandler import *
from consts import *

#precomputed values for datasets

#dataset1 pions
dataset0_params ={
'logit_mean' : -12.4783,
'logit_std' : 2.21267,
'logit_min': -13.81551,
'logit_max' :  0.9448,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,


'totalE_mean' : 0.,
'totalE_std' : 1.0,
'layers_mean' : 0.,
'layers_std' : 1.0,

'layerN_logit_mean' : 0., 
'layerN_logit_std': 1.0,


'qt' : 'qts/dset1_pions_quantile_transform.gz',
}


#dataset1 pions, no geom reshaping 
dataset0_origShape_params ={
'logit_mean' : -11.7610,
'logit_std' : 2.84317,
'logit_min': -13.81551,
'logit_max' :  0.2554,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,



'totalE_mean' : 0.2359,
'totalE_std' : 0.08255,
'layers_mean' : -4.9742,
'layers_std' : 4.89629,


'layerN_logit_mean' : -8.1456, 
'layerN_logit_std': 5.43684,


'qt' : None,
}




#dataset1 photons
dataset1_params ={
'logit_mean' : -12.1444,
'logit_std' : 2.45056,
'logit_min': -13.81551,
'logit_max' :  -1.6615,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,



'totalE_mean' : 0.,
'totalE_std' : 1.0,
'layers_mean' : 0.,
'layers_std' : 1.0,

'layerN_logit_mean' : 0., 
'layerN_logit_std': 1.0,

'qt' : 'qts/dset1_photons_quantile_transform.gz',
}


#dataset1 photons, no geom reshaping
dataset1_origShape_params ={
'logit_mean' : -9.9807,
'logit_std' : 3.14168,
'logit_min': -13.81551,
'logit_max' :  0.2554,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,


'totalE_mean' : 0.3123,
'totalE_std' : 0.02872,
'layers_mean' : -4.9023,
'layers_std' : 5.17364,

'layerN_logit_mean' : -8.2197, 
'layerN_logit_std': 4.18973,

'qt' : None,
}

dataset2_params = {
'logit_mean' : -12.8564,
'logit_std' : 1.9123,
'logit_min': -13.8155,
'logit_max' :  0.1153,

'log_mean' : -17.5451,
'log_std' : 4.4086,
'log_min' : -20.0,
'log_max' :  -0.6372,


'totalE_mean' : 0.3926,
'totalE_std' : 0.05546,
'layers_mean' : -6.35551,
'layers_std' : 3.90699,
#'layers_mean' : -7.1796,
#'layers_std' : 5.53775,

'layerN_logit_mean' : -9.2022, 
'layerN_logit_std': 5.69892,

'qt' : 'qts/dset2_quantile_transform.gz',
}


dataset3_params = {
'logit_mean' : -13.4753,
'logit_std' : 1.1070,
'logit_min': -13.81551,
'logit_max' :  0.2909,

'log_mean' : -1.1245,
'log_std' : 3.3451,
'log_min' : -18.6905,
'log_max' : 0.0,


'totalE_mean' : 0.,
'totalE_std' : 1.0,
'layers_mean' : 0.,
'layers_std' : 1.0,


'qt' : 'qts/dset3_quantile_transform.gz',
}
dataset_params = {
        0: dataset0_params, 
        1: dataset1_params, 
        2:dataset2_params, 
        3:dataset3_params,
        10: dataset0_origShape_params,
        11: dataset1_origShape_params,
        }

#use tqdm if local, skip if batch job
import sys
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable

def split_data_np(data, frac=0.8):
    np.random.shuffle(data)
    split = int(frac * data.shape[0])
    train_data =data[:split]
    test_data = data[split:]
    return train_data,test_data




def reverse_logit(x, alpha = 1e-6):
    exp = np.exp(x)    
    o = exp/(1+exp)
    o = (o-alpha)/(1 - 2*alpha)
    return o


def logit(x, alpha = 1e-6):
    o = alpha + (1 - 2*alpha)*x
    o = np.ma.log(o/(1-o)).filled(0)    
    return o



def DataLoader(file_name,shape,emax,emin, nevts=-1,  max_deposit = 2, ecut = 0, logE=True, showerMap = 'log-norm', nholdout = 0, from_end = False, dataset_num = 2, orig_shape = False,
        evt_start = 0):

    with h5.File(file_name,"r") as h5f:
        #holdout events for testing
        if(nevts == -1 and nholdout > 0): nevts = -(nholdout)
        end = evt_start + int(nevts)
        if(from_end):
            evt_start = -int(nevts)
            end = None
        if(end == -1): end = None 
        print("Event start, stop: ", evt_start, end)
        e = h5f['incident_energies'][evt_start:end].astype(np.float32)/1000.0
        shower = h5f['showers'][evt_start:end].astype(np.float32)/1000.0

        
    e = np.reshape(e,(-1,1))


    shower_preprocessed, layerE_preprocessed = preprocess_shower(shower, e, shape, showerMap, dataset_num = dataset_num, orig_shape = orig_shape, ecut = ecut, max_deposit=max_deposit)

    if logE:        
        E_preprocessed = np.log10(e/emin)/np.log10(emax/emin)
    else:
        E_preprocessed = (e-emin)/(emax-emin)

    return shower_preprocessed, E_preprocessed , layerE_preprocessed


    
def preprocess_shower(shower, e, shape, showerMap = 'log-norm', dataset_num = 2, orig_shape = False, ecut = 0, max_deposit = 2):

    if(dataset_num == 1): 
        binning_file = "binning_dataset_1_photons.xml"
        bins = XMLHandler("photon", binning_file)
    elif(dataset_num == 0): 
        binning_file = "binning_dataset_1_pions.xml"
        bins = XMLHandler("pion", binning_file)

    if(dataset_num  <= 1 and not orig_shape): 
        g = GeomConverter(bins)
        shower = g.convert(g.reshape(shower))
    elif(not orig_shape):
        shower = shower.reshape(shape)



    if(dataset_num > 3 or dataset_num <0 ): 
        print("Invalid dataset %i!" % dataset_num)
        exit(1)

    if(orig_shape and dataset_num <= 1): dataset_num +=10 

    print('dset', dataset_num)

    c = dataset_params[dataset_num]

    if('quantile' in showerMap and ecut > 0):
        np.random.seed(123)
        noise = (ecut/3) * np.random.rand(*shower.shape)
        shower +=  noise


    alpha = 1e-6
    per_layer_norm = False

    layerE = None
    prefix = ""
    if('layer' in showerMap):
        eshape = (-1, *(1,)*(len(shower.shape) -1))
        shower = np.ma.divide(shower, (max_deposit*e.reshape(eshape)))
        #regress total deposited energy and fraction in each layer
        if(dataset_num % 10 > 1 or not orig_shape):
            layers = np.sum(shower,(3,4),keepdims=True)
            totalE = np.sum(shower, (2,3,4), keepdims = True)
            if(per_layer_norm): shower = np.ma.divide(shower,layers)
            shower = np.reshape(shower,(shower.shape[0],-1))

        else:
            #use XML handler to deal with irregular binning of layers for dataset 1
            boundaries = np.unique(bins.GetBinEdges())
            layers = np.zeros((shower.shape[0], boundaries.shape[0]-1), dtype = np.float32)

            totalE = np.sum(shower, 1, keepdims = True)
            for idx in range(boundaries.shape[0] -1):
                layers[:,idx] = np.sum(shower[:,boundaries[idx]:boundaries[idx+1]], 1)
                if(per_layer_norm): shower[:,boundaries[idx]:boundaries[idx+1]] = np.ma.divide(shower[:,boundaries[idx]:boundaries[idx+1]], layers[:,idx:idx+1])


        #only logit transform for layers
        layer_alpha = 1e-6
        layers = np.ma.divide(layers,totalE)
        layers = logit(layers)


        layers = (layers - c['layers_mean']) / c['layers_std']
        totalE = (totalE - c['totalE_mean']) / c['totalE_std']
        #append totalE to layerE array
        totalE = np.reshape(totalE, (totalE.shape[0], 1))
        layers = np.squeeze(layers)
        layerE = np.concatenate((totalE,layers), axis = 1)

        if(per_layer_norm): prefix = "layerN_"
    else:

        shower = np.reshape(shower,(shower.shape[0],-1))
        shower = shower/(max_deposit*e)





    if('logit' in showerMap):
        shower = logit(shower)

        if('norm' in showerMap): shower = (shower - c[prefix +'logit_mean']) / c[prefix+'logit_std']
        elif('scaled' in showerMap): shower = 2.0 * (shower - c['logit_min']) / (c['logit_max'] - c['logit_min']) - 1.0

    elif('log' in showerMap):
        eps = 1e-8
        shower = np.ma.log(shower).filled(c['log_min'])
        if('norm' in showerMap): shower = (shower - c[prefix+'log_mean']) / c[prefix+'log_std']
        elif('scaled' in showerMap):  shower = 2.0 * (shower - c[prefix+'log_min']) / (c[prefix+'log_max'] - c[prefix+'log_min']) - 1.0


    if('quantile' in showerMap and c[prefix+'qt'] is not None):
        print("Loading quantile transform from %s" % c['qt'])
        qt = joblib.load(c['qt'])
        shape = shower.shape
        shower = qt.transform(shower.reshape(-1,1)).reshape(shower.shape)
        

    return shower,layerE

        

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


def ReverseNorm(voxels,e,shape,emax,emin,max_deposit=2,logE=True, layerE = None, showerMap ='log', dataset_num = 2, orig_shape = False, ecut = 0.):
    '''Revert the transformations applied to the training set'''

    if(dataset_num > 3 or dataset_num <0 ): 
        print("Invalid dataset %i!" % dataset_num)
        exit(1)

    if(dataset_num == 1): 
        binning_file = "binning_dataset_1_photons.xml"
        bins = XMLHandler("photon", binning_file)
    elif(dataset_num == 0): 
        binning_file = "binning_dataset_1_pions.xml"
        bins = XMLHandler("pion", binning_file)


    if(orig_shape and dataset_num <= 1): dataset_num +=10 
    print('dset', dataset_num)
    c = dataset_params[dataset_num]

    alpha = 1e-6
    if logE:
        energy = emin*(emax/emin)**e
    else:
        energy = emin + (emax-emin)*e

    prefix = ""
    #if('layer' in showerMap): prefix = "layerN_"

    if('quantile' in showerMap and c['qt'] is not None):
        print("Loading quantile transform from %s" % c['qt'])
        qt = joblib.load(c['qt'])
        shape = voxels.shape
        voxels = qt.inverse_transform(voxels.reshape(-1,1)).reshape(shape)

        
    if('logit' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c[prefix+'logit_std']) + c[prefix+'logit_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0) * 0.5 * (c[prefix+'logit_max'] - c[prefix+'logit_min']) + c[prefix+'logit_min']

        #avoid overflows
        #voxels = np.minimum(voxels, np.log(max_deposit/(1-max_deposit)))

        data = reverse_logit(voxels)

    elif('log' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c[prefix+'log_std']) + c[prefix+'log_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0) * 0.5 * (c[prefix+'log_max'] - c[prefix+'log_min']) + c[prefix+'log_min']

        voxels = np.minimum(voxels, np.log(max_deposit))


        data = np.exp(voxels)

    #Per layer energy normalization
    if('layer' in showerMap):
        assert(layerE is not None)
        totalE, layers = layerE[:,:1], layerE[:,1:]
        totalE = (totalE * c['totalE_std']) + c['totalE_mean']
        layers = (layers * c['layers_std']) + c['layers_mean']

        layers = reverse_logit(layers)

        #scale layer energies to total deposited energy
        layers /= np.sum(layers, axis = 1, keepdims = True)
        layers *= totalE


        data = np.squeeze(data)

        #remove voxels with negative energies so they don't mess up sums
        eps = 1e-6
        data[data < 0] = 0 
        #layers[layers < 0] = eps


        #Renormalize layer energies
        if(dataset_num%10 > 1 or not orig_shape):
            prev_layers = np.sum(data,(2,3),keepdims=True)
            layers = layers.reshape((-1,data.shape[1],1,1))
            rescale_facs =  layers / (prev_layers + 1e-10)
            #If layer is essential zero from base network or layer network, don't rescale
            rescale_facs[layers < eps] = 1.0
            rescale_facs[prev_layers < eps] = 1.0
            data *= rescale_facs
        else:
            boundaries = np.unique(bins.GetBinEdges())
            for idx in range(boundaries.shape[0] -1):
                prev_layer = np.sum(data[:,boundaries[idx]:boundaries[idx+1]], 1, keepdims=True)
                rescale_fac  = layers[:,idx:idx+1] / (prev_layer + 1e-10)
                rescale_fac[layers[:, idx:idx+1] < eps] = 1.0
                rescale_fac[prev_layer < eps] = 1.0
                data[:,boundaries[idx]:boundaries[idx+1]] *= rescale_fac
                    
                



    if(dataset_num > 1 or orig_shape): 
        data = data.reshape(voxels.shape[0],-1)*max_deposit*energy.reshape(-1,1)
    else:
        g = GeomConverter(bins)
        data = np.squeeze(data)
        data = g.unreshape(g.unconvert(data))*max_deposit*energy.reshape(-1,1)

    if('quantile' in showerMap and ecut > 0.):
        #subtact of avg of added noise
        data -= 0.5 * (ecut/3)

    if(ecut > 0): data[data < ecut ] = 0 #min from samples
    
    return data,energy
    





class GeomConverter:
    "Convert irregular geometry to regular one (ala CaloChallenge Dataset 1)"
    def __init__(self, bins = None, all_r_edges = None, lay_r_edges = None, alpha_out = 1, lay_alphas = None):

        self.layer_boundaries = []
        self.bins = None

        #init from binning
        if(bins is not None):
            

            self.layer_boundaries = np.unique(bins.GetBinEdges())
            rel_layers = bins.GetRelevantLayers()
            lay_alphas = [len(bins.alphaListPerLayer[idx][0]) for idx, redge in enumerate(bins.r_edges) if len(redge) > 1]
            alpha_out = np.amax(lay_alphas)


            all_r_edges = []

            lay_r_edges = [bins.r_edges[l] for l in rel_layers]
            for ilay in range(len(lay_r_edges)):
                for r_edge in lay_r_edges[ilay]:
                    all_r_edges.append(r_edge)
            all_r_edges = torch.unique(torch.FloatTensor(all_r_edges))

        self.all_r_edges = all_r_edges
        self.lay_r_edges = lay_r_edges
        self.alpha_out = alpha_out
        self.lay_alphas = lay_alphas
        self.num_layers = len(self.lay_r_edges)


        self.all_r_areas = (all_r_edges[1:]**2 - all_r_edges[:-1]**2)
        self.dim_r_out = len(all_r_edges) - 1
        self.weight_mats = []
        for ilay in range(len(lay_r_edges)):
            dim_in = len(lay_r_edges[ilay]) - 1
            lay = nn.Linear(dim_in, self.dim_r_out, bias = False)
            weight_mat = torch.zeros((self.dim_r_out, dim_in))
            for ir in range(dim_in):
                o_idx_start = torch.nonzero(self.all_r_edges == self.lay_r_edges[ilay][ir])[0][0]
                o_idx_stop = torch.nonzero(self.all_r_edges == self.lay_r_edges[ilay][ir + 1])[0][0]

                split_idxs = list(range(o_idx_start, o_idx_stop))
                orig_area = (self.lay_r_edges[ilay][ir+1]**2 - self.lay_r_edges[ilay][ir]**2)

                #split proportional to bin area
                weight_mat[split_idxs, ir] = self.all_r_areas[split_idxs]/orig_area

            self.weight_mats.append(weight_mat)



    def reshape(self, raw_shower):
        #convert to jagged array each of shape (N_shower, N_alpha, N_R)
        shower_reshape = []
        for idx in range(len(self.layer_boundaries)-1):
            data_reshaped = raw_shower[:,self.layer_boundaries[idx]:self.layer_boundaries[idx+1]].reshape(raw_shower.shape[0], int(self.lay_alphas[idx]), -1)
            shower_reshape.append(data_reshaped)
        return shower_reshape

    def unreshape(self, raw_shower):
        #convert jagged back to original flat format
        n_show = raw_shower[0].shape[0]
        out = torch.zeros((n_show, self.layer_boundaries[-1]))
        for idx in range(len(self.layer_boundaries)-1):
            out[:, self.layer_boundaries[idx]:self.layer_boundaries[idx+1]] = raw_shower[idx].reshape(n_show, -1)
        return out


    def convert(self, d):
        out = torch.zeros((len(d[0]), self.num_layers, self.alpha_out, self.dim_r_out))
        for i in range(len(d)):
            if(not isinstance(d[i], torch.FloatTensor)): d[i] = torch.FloatTensor(d[i])
            o = torch.einsum( '...ij,...j->...i', self.weight_mats[i], d[i])
            if(self.lay_alphas is not None):
                if(self.lay_alphas[i]  == 1):
                    #distribute evenly in phi
                    o = torch.repeat_interleave(o, self.alpha_out, dim = -2)/self.alpha_out
                elif(self.lay_alphas[i]  != self.alpha_out):
                    print("Num alpha bins for layer %i is %i. Don't know how to handle" % (i, self.lay_alphas[i]))
                    exit(1)
            out[:,i] = o
        return out


    def unconvert(self, d):
        out = []
        for i in range(self.num_layers):
            weight_mat_inv = torch.linalg.pinv(self.weight_mats[i])
            x = torch.FloatTensor(d[:,i])
            o = torch.einsum( '...ij,...j->...i', weight_mat_inv, x)

            if(self.lay_alphas is not None):
                if(self.lay_alphas[i]  == 1):
                    #Only works for converting 1 alpha bin into multiple, ok for dataset1 but maybe should generalize
                    o = torch.sum(o, dim = -2, keepdim = True)
                elif(self.lay_alphas[i]  != self.alpha_out):
                    print("Num alpha bins for layer %i is %i. Don't know how to handle" % (i, self.lay_alphas[i]))
                    exit(1)
            out.append(o)
        return out


