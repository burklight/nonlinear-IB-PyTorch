import torch
import torch.nn
import torch.nn.init
import torchvision
import argparse


def weight_init(m):
    '''
    This function is used to initialize the netwok weights
    '''

    if isinstance(m,torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)

def get_data():
    '''
    This function returns the training and validation set from MNIST
    '''

    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    validationset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    return trainset, validationset

def get_args():
    '''
    This function returns the arguments from terminal and set them to display
    '''

    parser = argparse.ArgumentParser(
        description = 'Run non-linear IB on MNIST dataset (with Pytorch)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--logs_dir', default = '../results/logs/',
        help = 'folder to output the logs')
    parser.add_argument('--figs_dir', default = '../results/figures/',
        help = 'folder to output the images')
    parser.add_argument('--models_dir', default = '../results/models/',
        help = 'folder to save the models')
    parser.add_argument('--n_epochs', type = int, default = 100,
        help = 'number of training epochs')
    parser.add_argument('--beta', type = float, default = 0.0,
        help = 'Lagrange multiplier (only for train_model)')
    parser.add_argument('--n_betas', type = int, default = 21,
        help = 'Number of Lagrange multipliers (only for study behavior)')
    parser.add_argument('--K', type = int, default = 2,
        help = 'Dimensionality of the bottleneck varaible')
    parser.add_argument('--logvar_kde', type = float, default = -1.0,
        help = 'initial log variance of the KDE estimator')
    parser.add_argument('--logvar_t', type = float, default = -1.0,
        help = 'initial log varaince of the bottleneck variable')
    parser.add_argument('--sgd_batch_size', type = int, default = 128,
        help = 'mini-batch size for the SGD on the error')
    parser.add_argument('--mi_batch_size', type = int, default = 1000,
        help = 'mini-batch size for the I(X;T) estimation')
    parser.add_argument('--same_batch', action = 'store_true', default = False,
        help = 'use the same mini-batch for the SGD on the error and I(X;T) estimation')
    parser.add_argument('--optimizer_name', choices = ['sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'asgd'], default = 'adam',
        help = 'optimizer')
    parser.add_argument('--learning_rate', type = float, default = 0.0001,
        help = 'initial learning rate')
    parser.add_argument('--learning_rate_drop', type = float, default = 0.6,
        help = 'learning rate decay rate (step LR every learning_rate_steps)')
    parser.add_argument('--learning_rate_steps', type = int, default = 10,
        help = 'number of steps (epochs) before decaying the learning rate')
    parser.add_argument('--train_logvar_t', action = 'store_true', default = False,
        help = 'train the log(variance) of the bottleneck variable')
    parser.add_argument('--eval_rate', type = int, default = 20,
        help = 'evaluate I(X;T), I(T;Y) and accuracies every eval_rate epochs')
    parser.add_argument('--visualize', action = 'store_true', default = False,
        help = 'visualize the results every eval_rate epochs')
    parser.add_argument('--verbose', action = 'store_true', default = False,
        help = 'report the results every eval_rate epochs')

    return parser.parse_args()
