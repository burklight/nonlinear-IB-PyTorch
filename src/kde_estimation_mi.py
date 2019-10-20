import torch
import math
import numpy as np
import scipy.special

################################################################################
## KDE estimation of the information (PyTorch)
################################################################################

def compute_distances(x):
    '''
    Computes the distance matrix for the KDE Entropy estimation:
    - x (Tensor) : array of functions to compute the distances matrix from
    '''

    x_norm = (x**2).sum(1).view(-1,1)
    x_t = torch.transpose(x,0,1)
    x_t_norm = x_norm.view(1,-1)
    dist = x_norm + x_t_norm - 2.0*torch.mm(x,x_t)
    dist = torch.clamp(dist,0,np.inf)

    return dist

def KDE_IXT_estimation(logvar_t, mean_t):
    '''
    Computes the MI estimation of X and T. Parameters:
    - logvar_t (float) : log(var) of the bottleneck variable 
    - mean_t (Tensor) : deterministic transformation of the input 
    '''

    n_batch, d = mean_t.shape
    var = torch.exp(logvar_t) + 1e-10 # to avoid 0's in the log

    # calculation of the constant
    normalization_constant = math.log(n_batch)

    # calculation of the elements contribution
    dist = compute_distances(mean_t)
    distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var,dim=1))

    # mutual information calculation (natts)
    I_XT = normalization_constant + distance_contribution 

    return I_XT