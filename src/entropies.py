import torch
import autograd.numpy as anp
import numpy as np
import scipy.special

################################################################################
## KDE estimation of the variance of the MoG (autograd)
################################################################################

def compute_distances_loo(x):
    '''
    Compute the distance matrix for the Leave-One-Out negative log likelihood
    Parameters:
    - x (numpy array) : array of functions to compute the distances matrix from
    '''

    x_norm = np.sum(x**2,axis=1).reshape(-1,1)
    x_t = x.T
    x_t_norm = x_norm.reshape(1,-1)
    dist = x_norm + x_t_norm - 2.0*x.dot(x_t)
    dist = dist + 10e20*np.eye(dist.shape[0]) # deprecate diagonals
    dist = np.clip(dist,0,np.inf)

    return dist

def log_sum_exp_loo(x):
    '''
    Log sum exp function with the ability to compute automatic gradients
    - x (numpy array)
    '''

    x_max = anp.max(x,axis=0)
    with anp.errstate(divide='ignore'):
        out = anp.log(anp.sum(anp.exp(x - x_max),axis=0))
    out += x_max

    return out

def KDE_loo_neg_log_likelihood(logvar_kde, dist, n_batch, d):
    '''
    This function computes the Leave-One-Out negative log-likelihood of the
    training data.
    Parameters:
    - logvar_kde (float) : log(var) of the kernel density estimation
    - dist (numpy array) : matrix of distances of the deterministic function of the input
    - n_batch (int) : number of elements in that mini-batch
    - d (int) : dimensionality of the deterministic function of the input
    '''

    var_kde = anp.exp(logvar_kde) + 1e-10 # to avoid 0's in the logs
    norm_constant = - anp.log(n_batch - 1)
    gaussian_constant = - 0.5 * d * anp.log(2.0*anp.pi*var_kde)
    gaussian_distance = log_sum_exp_loo(- 0.5 * dist / var_kde)

    log_likelihoods = norm_constant + gaussian_constant + gaussian_distance
    neg_log_likelihood = - anp.mean(log_likelihoods)

    return neg_log_likelihood

################################################################################
## KDE estimation of the information (PyTorch)
################################################################################

def compute_distances(x):
    '''
    Compute the distance matrix for the KDE Entropy estimation:
    - x (Tensor) : array of functions to compute the distances matrix from
    '''

    x_norm = (x**2).sum(1).view(-1,1)
    x_t = torch.transpose(x,0,1)
    x_t_norm = x_norm.view(1,-1)
    dist = x_norm + x_t_norm - 2.0*torch.mm(x,x_t)
    dist = torch.clamp(dist,0,np.inf)

    return dist


def KDE_entropy_t_given_x(logvar_t, d):
    '''
    Computes the KDE estimation of HT_given_X
    Parameters:
    - logvar_t (float) : log(var) of the bottleneck variable
    - d (int) : dimensionality of the bottleneck variable
    '''

    # Entropy calculation (nats)
    H_T_given_X = 0.5 * d * (np.log(2.0*np.pi) + logvar_t)

    return H_T_given_X

def KDE_entropy_t(logvar_t, logvar_kde, mean_t):
    '''
    Computes the KDE etimation of HT
    Parameters:
    - logvar_t (float) : log(var) of the bottleneck variable
    - logvar_kde (float) : log(var) of the kernel density estimation
    - mean_t (Tensor) : deterministic transformation of the input 
    '''

    n_batch, d = mean_t.shape
    var = torch.exp(logvar_t) + np.exp(logvar_kde) + 1e-10 # to avoid 0's in the log

    # calculation of the constants
    gaussian_constant = 0.5 * d * torch.log(2.0*np.pi*var)
    normalization_constant = np.log(n_batch)

    # calculation of the elements contribution
    dist = compute_distances(mean_t)
    distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var,dim=1))

    # Entropy calculation (nats)
    H_T = normalization_constant + distance_contribution + gaussian_constant

    return H_T
