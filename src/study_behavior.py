from nonlinearIB import NonlinearIB
from utils import get_data
from utils import get_args
from visualization import plot_behavior
import torch
import os
import datetime
import numpy as np
import multiprocessing
import math

torch.set_num_threads(1)

# Obtain the arguments
args = get_args()

# Obtain the data
dataset_name = args.dataset
trainset, validationset, testset = get_data(dataset=dataset_name,percentage_validation=0.2)
if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
    n_x = 784
    n_y = 10
    HY = np.log2(n_y)
    maxIXY = HY
    problem_type = 'classification'
elif dataset_name == 'california_housing':
    n_x = 8
    n_y = 1
    varY = torch.var(trainset.targets)
    HY = 0.5 * math.log(varY.item()*2.0*math.pi*math.e) / np.log(2)
    maxIXY = 0.72785 / np.log(2) # Estimation by training with only the cross entropy and getting the result after training
    problem_type = 'regression'

# Create the folders
logs_dir = os.path.join(args.logs_dir,dataset_name) + '/'
figs_dir = os.path.join(args.figs_dir,dataset_name) + '/'
models_dir = os.path.join(args.models_dir,dataset_name) + '/'
os.makedirs(logs_dir) if not os.path.exists(logs_dir) else None
os.makedirs(figs_dir) if not os.path.exists(figs_dir) else None
os.makedirs(models_dir) if not os.path.exists(models_dir) else None

# Range of the lagrange multiplier
betas = np.linspace(args.beta_lim_min,args.beta_lim_max,args.n_betas)

# Function for multiprocessing
def train_and_save(beta):

    print("--- Studying Non-Linear IB behavior with beta = " + str(round(beta,3)) + " ---")
    # Train the network
    nonlinear_IB = NonlinearIB(n_x = n_x, n_y = n_y, problem_type = problem_type, K = args.K, beta = beta, 
        logvar_t = args.logvar_t, train_logvar_t = args.train_logvar_t, hfunc = args.hfunc, param = args.hfunc_param)
    nonlinear_IB.fit(trainset, validationset, testset, n_epochs = args.n_epochs, learning_rate = args.learning_rate,
        learning_rate_drop = args.learning_rate_drop, learning_rate_steps = args.learning_rate_steps, sgd_batch_size = args.sgd_batch_size,
        early_stopping_lim = args.early_stopping_lim, eval_rate = args.eval_rate, optimizer_name = args.optimizer_name, verbose = args.verbose, 
        visualization = args.visualize, logs_dir = logs_dir, figs_dir = figs_dir)

    # Save the network
    name_base = 'dimT-'+str(args.K) + '-beta-' + ('%.3f' % args.beta).replace('.',',') + '--hfunc-' \
        + args.hfunc + '--param-' + ('%.3f' % args.hfunc_param).replace('.',',') 
    torch.save(nonlinear_IB.state_dict(), args.models_dir + name_base + 'model')


# For all betas...
with multiprocessing.Pool(4) as p: 
    p.map(train_and_save, betas)

# Visualize the comparison
plot_behavior(logs_dir,figs_dir,args.K,betas,args.hfunc,args.hfunc_param,maxIXY)
