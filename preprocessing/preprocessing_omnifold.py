import os
import h5py as h5
import numpy as np
import json, yaml
import corner
import energyflow as ef

"""
    for calochallenge low-level feature preprocessing, based on the code from omniFold and SBUnfold

    input:
        - files donwloaded from zenodo
    output:
        - numpy array stored in .npy
    
    eg usage:
    
    train_gen, train_sim, test_gen,test_sim = utils.DataLoader('Pythia26',json_path = '../JSON')
    save them into files

        
"""

def DataLoader(sample_name,
               N_t=300000,N_v=200000,
               cache_dir="../data/",json_path='JSON'):

    datasets = {sample_name: ef.zjets_delphes.load(sample_name, num_data=N_t+N_v,cache_dir=cache_dir,exclude_keys=['particles'])}
    feature_names = ['widths','mults','sdms','zgs','tau2s']
    gen_features = [datasets[sample_name]['gen_jets'][:,3]]
    sim_features = [datasets[sample_name]['sim_jets'][:,3]]

    for feature in feature_names:
        gen_features.append(datasets[sample_name]['gen_'+feature])
        sim_features.append(datasets[sample_name]['sim_'+feature])

    gen_features = np.stack(gen_features,-1)
    sim_features = np.stack(sim_features,-1)
    
    #ln rho
    gen_features[:,3] = 2*np.ma.log(np.ma.divide(gen_features[:,3],datasets[sample_name]['gen_jets'][:,0]).filled(0)).filled(0)
    sim_features[:,3] = 2*np.ma.log(np.ma.divide(sim_features[:,3],datasets[sample_name]['sim_jets'][:,0]).filled(0)).filled(0)
    
    #tau2
    gen_features[:,5] = gen_features[:,5]/(10**-50 + gen_features[:,1])
    sim_features[:,5] = sim_features[:,5]/(10**-50 + sim_features[:,1])

    #Standardize
    gen_features = ApplyPreprocessing(gen_features,'gen_features.json',json_path)
    sim_features = ApplyPreprocessing(sim_features,'sim_features.json',json_path)

    train_gen = []
    train_sim = []
    
    for i in range(0,N_t):
        
        if (gen_features[i,0] < 4) and (sim_features[i,0] < 4) and (gen_features[i,1] < 3) and (sim_features[i,1] < 3) and \
        (gen_features[i,2] < 3) and (sim_features[i,2] < 3) and (np.abs(gen_features[i,3]) < 2) and (np.abs(sim_features[i,3]) < 2) and (np.abs(gen_features[i,4]) < 1) and (np.abs(sim_features[i,4]) < 1):
            train_gen.append(gen_features[i])
            train_sim.append(sim_features[i])

    test_gen = []
    test_sim = []
    
    for i in range(N_t,N_t+N_v):
        
        if (gen_features[i,0] < 4) and (sim_features[i,0] < 4) and (gen_features[i,1] < 3) and (sim_features[i,1] < 3) and \
        (gen_features[i,2] < 3) and (sim_features[i,2] < 3) and (np.abs(gen_features[i,3]) < 2) and (np.abs(sim_features[i,3]) < 2) and (np.abs(gen_features[i,4]) < 1) and (np.abs(sim_features[i,4]) < 1):
            test_gen.append(gen_features[i])
            test_sim.append(sim_features[i])
    
    train_gen = np.array(train_gen)
    train_sim = np.array(train_sim)
    test_gen = np.array(test_gen)
    test_sim = np.array(test_sim)
    

    return train_gen, train_sim, test_gen,test_sim
    
def CalcPreprocessing(data,fname,base_folder):
    '''Apply data preprocessing'''
    
    data_dict = {}
    mean = np.average(data,axis=0)
    std = np.std(data,axis=0)
    data_dict['mean']=mean.tolist()
    data_dict['std']=std.tolist()
    data_dict['min']=np.min(data,0).tolist()
    data_dict['max']=np.max(data,0).tolist()    
    SaveJson(fname,data_dict,base_folder)



def ApplyPreprocessing(data,fname,base_folder): 
    data_dict = LoadJson(fname,base_folder)
    data = (np.ma.divide((data-data_dict['mean']),data_dict['std']).filled(0)).astype(np.float32)
    return data


def ReversePreprocessing(data,fname,base_folder):
    data_dict = LoadJson(fname,base_folder)
    data = data * data_dict['std'] + data_dict['mean']
    return data




def SaveJson(save_file,data,base_folder='JSON'):
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    
    with open(os.path.join(base_folder,save_file),'w') as f:
        json.dump(data, f)

    
def LoadJson(file_name,base_folder='JSON'):
    import json,yaml
    JSONPATH = os.path.join(base_folder,file_name)
    return yaml.safe_load(open(JSONPATH))
