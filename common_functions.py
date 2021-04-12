import torch
import numpy as np
import random
import os
import datetime
import time
import pickle
import bz2
from typing import OrderedDict

def lazy_load_model_state(model, filename, fuzy_match=True):
        # will ignore missing or additional layer names. Also, if size missmatch it will ignore! 
        # if fuzzy_match, matches the end of the 

        file_model_state_dict = torch.load(filename)
        to_load_dict = OrderedDict()


        model_state_dict = model.state_dict()

        for key in file_model_state_dict:

            if key in model_state_dict.keys():
                # key exists
                if file_model_state_dict[key].shape == model_state_dict[key].shape:
                    # happy days it matches in name and shape, copy it over
                    to_load_dict[key] = file_model_state_dict[key]
                else:
                    # doesnt match size, dont copy it over
                    pass
            else:
                # doesnt exist, but check if we are doing fuzy
                if fuzy_match:
                    
                    matching_keys = [k for k in model_state_dict.keys() if k.endswith(key)]

                    for l in matching_keys:
                        #match keys, again need to check size
                        if file_model_state_dict[key].shape == model_state_dict[l].shape:
                            to_load_dict[l] = file_model_state_dict[key]


        model.load_state_dict(to_load_dict, strict=False)
        model.eval()

        return model

def ensure_reproducibility(seed=None):
    '''
    Sets all the random seeds to a specified seed.
    :param seed: a seed to set. If none, sets the seed as that specified in the defaults file
    :return: none
    '''

    if seed is None:
        seed = 6000

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)   << doesn't work with multiple workers for some reason. https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

def save_model_state(model, filename):
    torch.save(model.state_dict(), filename)
    return True

def load_model_state(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def pickle_object(obj, path, zip_file=True):
    if zip_file:
        f = bz2.open(path, 'wb')
    else:
        f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()
    
def load_pickled_object(path, zip_file=True):
    if zip_file:
        f = bz2.open(path, 'rb')
    else:
        f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

if __name__ == '__main__':
    pass