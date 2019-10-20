import torch
import torch.nn
import torch.nn.init
import torchvision
import argparse
import PIL.Image
import sklearn.datasets
import random

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

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,dataset,idxs):
        self.data = dataset.data[idxs]
        self.targets = dataset.targets[idxs]
        self.transform = dataset.transform
    
    def __getitem__(self,index):
        img, target = self.data[index], self.targets[index]
        img, target = PIL.Image.fromarray(img.numpy(), mode='L'), int(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)

def split_dataset(dataset,percentage_validation):
    '''
    This function splits a dataset into training and validation. Parameters:
    - dataset (torch.utils.data.Dataset) : dataset to split 
    - percentage_validation (float) : percentage of the original training set sent to validation 
    ''' 

    
    splitpoint = int(len(dataset) * (1.0 - percentage_validation))
    idxs = torch.randperm(len(dataset))
    trainset = CustomDataset(dataset,idxs[:splitpoint])
    validationset = CustomDataset(dataset,idxs[splitpoint:])
    return trainset, validationset

def get_mnist(percentage_validation=0.2):
    ''' 
    This function returns the MNIST dataset in training, validation, test splits. Parameters:
    - percentage_validation (float) : Percentage of the original training set sent to validation 
    '''

    trainset_original = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    
    trainset, validationset = split_dataset(trainset_original, percentage_validation)
    return trainset, validationset, testset

def get_fashion_mnist(percentage_validation=0.2):
    ''' 
    This function returns the MNIST dataset in training, validation, test splits. Parameters:
    - percentage_validation (float) : Percentage of the original training set sent to validation 
    '''

    trainset_original = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    
    trainset, validationset = split_dataset(trainset_original, percentage_validation)
    return trainset, validationset, testset

class DatasetRegression(torch.utils.data.Dataset):

    def __init__(self,X,Y):
        self.data = X
        self.targets = Y
    
    def __getitem__(self,index):
        data, target = self.data[index], self.targets[index]
        return data, target
    
    def __len__(self):
        return len(self.targets)

def get_california_housing(percentage_validation=0.2,percentage_test=0.2):



    X, Y = sklearn.datasets.fetch_california_housing(data_home='../data/CaliforniaHouring/', \
        download_if_missing=True,return_X_y=True)
    
    # We remove the houses with prices higher than 500,000 dollars
    idx_drop = Y >= 5
    X, Y = X[~idx_drop], Y[~idx_drop]

    # We shuffle the inputs and outputs before assigning train/val/test 
    tmp = list(zip(X,Y))
    random.shuffle(tmp)
    X, Y = zip(*tmp)
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)

    # Split between training / validation / testing
    splitpoint_test = int(len(Y) * (1.0 - percentage_test))
    splitpoint_validation = int(splitpoint_test * (1.0 - percentage_validation))
    X_train, Y_train = X[:splitpoint_validation], Y[:splitpoint_validation]
    X_validation, Y_validation = X[splitpoint_validation:splitpoint_test], Y[splitpoint_validation:splitpoint_test]
    X_test, Y_test = X[splitpoint_test:], Y[splitpoint_test:]

    # Generate and return the datasets
    trainset = DatasetRegression(X_train,Y_train)
    validationset = DatasetRegression(X_validation,Y_validation)
    testset = DatasetRegression(X_test,Y_test)
    return trainset,validationset, testset


def get_data(dataset='mnist',percentage_validation=0.2):
    '''
    This function returns the training and validation set from MNIST
    '''

    if dataset == 'mnist':
        return get_mnist(percentage_validation)
    elif dataset == 'fashion_mnist':
        return get_fashion_mnist(percentage_validation)
    elif dataset == 'california_housing':
        return get_california_housing(percentage_validation)

def get_args():
    '''
    This function returns the arguments from terminal and set them to display
    '''

    parser = argparse.ArgumentParser(
        description = 'Run nonlinear IB (with Pytorch)',
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
    parser.add_argument('--beta_lim_min', type = float, default = 0.0,
        help = 'minimum value of beta for the study of the behavior')
    parser.add_argument('--beta_lim_max', type = float, default = 1.0,
        help = 'maximum value of beta for the study of the behavior')  
    parser.add_argument('--hfunc', choices = ['exp', 'pow', 'none'], default = 'exp',
        help = 'Monotonically increasing, strictly convex function for the Lagrangian')
    parser.add_argument('--hfunc_param', type = float, default = 1.0,
        help = 'Parameter of the h function')
    parser.add_argument('--n_betas', type = int, default = 21,
        help = 'Number of Lagrange multipliers (only for study behavior)')
    parser.add_argument('--K', type = int, default = 2,
        help = 'Dimensionality of the bottleneck varaible')
    parser.add_argument('--logvar_t', type = float, default = 0.0,
        help = 'initial log varaince of the bottleneck variable')
    parser.add_argument('--sgd_batch_size', type = int, default = 256,
        help = 'mini-batch size for the SGD on the error')
    parser.add_argument('--early_stopping_lim', type = int, default = 20,
        help = 'early stopping limit for non improvement')
    parser.add_argument('--dataset', choices = ['mnist', 'fashion_mnist', 'california_housing'], default = 'mnist',
        help = 'dataset where to run the experiments. Classification: MNIST or Fashion MNIST. Regression: California housing.')
    parser.add_argument('--optimizer_name', choices = ['sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'asgd'], default = 'adam',
        help = 'optimizer')
    parser.add_argument('--learning_rate', type = float, default = 0.001,
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
