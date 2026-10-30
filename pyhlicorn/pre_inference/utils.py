import os
from itertools import islice

import numpy as np
import pandas as pd

def gene_specific_generator(gene_list, gene_disc_exp, gene_num_exp):
    assert len(gene_list) == len(gene_disc_exp) == len(gene_num_exp)
    for i in range(len(gene_list)):
        yield (gene_list[i], gene_disc_exp[i], gene_num_exp[i])
    
def sharedMemory(data : np.ndarray, data_name : str):
    
    pth = os.path.abspath(os.path.join(__file__, "../../temp"))
    
    try:
        os.mkdir(pth)
    except FileExistsError:
        pass
    
    output_filename_memmap = os.path.join(pth, data_name)
    
    output = np.memmap(
        output_filename_memmap, dtype=data.dtype, shape=data.shape, mode="w+"
    )
    
    output[:] = data[:]
    
    output.flush()
    
    return (output_filename_memmap, data.dtype, data.shape)


def load_shared_memmap(filename, dtype, shape):
    return np.memmap(filename, dtype=dtype, shape=shape, mode='r')


