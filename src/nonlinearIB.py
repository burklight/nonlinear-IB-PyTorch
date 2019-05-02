import torch
import math
import autograd
import scipy.optimize
from progressbar import progressbar
import numpy as np
from network import nlIB_network
from entropies import compute_distances_loo
from entropies import KDE_entropy_t
from entropies import KDE_entropy_t_given_x
from entropies import KDE_loo_neg_log_likelihood
from visualization import plot_results
from visualization import init_results_visualization
import matplotlib.pyplot as plt

class NonlinearIB(torch.nn.Module):
    '''
    Implementation of the Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    '''

    def __init__(self,n_x,n_y,K,beta,logvar_t=-1.0,logvar_kde=-1.0,\
        train_logvar_t=False):
        super(NonlinearIB,self).__init__()

        self.HY = np.log(n_y) # in natts

        self.K = K
        self.beta = beta
        self.logvar_kde = logvar_kde
        self.kde_objective = KDE_loo_neg_log_likelihood
        self.kde_jacobian = autograd.grad(KDE_loo_neg_log_likelihood)

        self.train_logvar_t = train_logvar_t
        self.network = nlIB_network(K,n_x,n_y,logvar_t,self.train_logvar_t)

    def update_logvar_kde(self, mean_t):
        '''
        Updates the log(var) of the kernel density estimation.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        mean_t = mean_t.detach().numpy()
        n_batch, d = mean_t.shape
        dist = compute_distances_loo(mean_t)
        self.logvar_kde = scipy.optimize.minimize(fun=self.kde_objective,
            x0=self.logvar_kde, jac=self.kde_jacobian, args=(dist,n_batch,d)).x[0]

    def get_IXT(self,mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        HT = KDE_entropy_t(self.network.logvar_t,self.logvar_kde,mean_t) # in natts
        HT_given_X = KDE_entropy_t_given_x(self.network.logvar_t,self.K) # in natts
        self.IXT = (HT - HT_given_X) / np.log(2) # in bits
        return self.IXT

    def get_ITY(self,logits_y,y):
        '''
        Obtains the mutual information between the bottleneck variable and the output.
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        ce = torch.nn.CrossEntropyLoss()
        HY_given_T = ce(logits_y,y)
        self.ITY = (self.HY - HY_given_T) / np.log(2) # in bits
        return self.ITY

    def evaluate(self,logits_y,y):
        '''
        Evauluates the performance of the model (accuracy)
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        with torch.no_grad():
            y_hat = y.eq(torch.max(logits_y,dim=1)[1])
            accuracy = torch.mean(y_hat.float())
        return accuracy

    def fit(self,trainset,validationset,n_epochs=200,learning_rate=0.0001,\
        learning_rate_drop=0.6,learning_rate_steps=10, sgd_batch_size=128,mi_batch_size=1000, \
        same_batch=True,eval_rate=20,optimizer_name='adam',verbose=True,visualization=True,
        logs_dir='.',figs_dir='.'):
        '''
        Trains the model with the training set and evaluates with the validation one.
        Parameters:
        - trainset (PyTorch Dataset) : Training dataset
        - validationset (PyTorch Dataset) : Validation dataset
        - n_epochs (int) : number of training epochs
        - learning_rate (float) : initial learning rate
        - learning_rate_drop (float) : multicative learning decay factor
        - learning_rate_steps (int) : number of steps before decaying the learning rate
        - sgd_batch_size (int) : size of the SGD mini-batch
        - mi_batch_size (int) : size of the MI estimation mini-batch
        - same_batch (bool) : if True, SGD and MI use the same mini-batch
        - eval_rate (int) : the model is evaluated every eval_rate epochs
        - verbose (bool) : if True, the evaluation is reported
        - visualization (bool) : if True, the evaluation is shown
        - logs_dir (str) : path for the storage of the evaluation
        - figs_dir (str) : path for the storage of the images of the evaluation
        '''

        # Definition of the training and validation losses, accuracies and MI
        report = 0
        n_reports = math.ceil(n_epochs / eval_rate)
        train_loss = np.zeros(n_reports)
        validation_loss = np.zeros(n_reports)
        train_accuracy = np.zeros(n_reports)
        validation_accuracy = np.zeros(n_reports)
        train_IXT = np.zeros(n_reports)
        train_ITY = np.zeros(n_reports)
        validation_IXT = np.zeros(n_reports)
        validation_ITY = np.zeros(n_reports)
        epochs = np.zeros(n_reports)

        # Data Loader
        n_sgd_batches = math.floor(len(trainset) / sgd_batch_size)
        sgd_train_loader = torch.utils.data.DataLoader(trainset, \
            batch_size=sgd_batch_size,shuffle=True)
        if not same_batch:
            n_mi_batches = math.floor(len(trainset) / mi_batch_size)
            mi_train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=mi_batch_size,shuffle=True)
            mi_train_batches = enumerate(mi_train_loader)
        validation_loader = torch.utils.data.DataLoader(validationset, \
            batch_size=len(validationset),shuffle=False)

        # Prepare visualization
        if visualization:
            visualization_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=10000,shuffle=False) # Only for visualization
            fig, ax = init_results_visualization(self.K)

        # Prepare name for figures and logs
        name_base = "K-" + str(self.K) + "-B-" + str(round(self.beta,3)).replace('.', '-') \
            + "-Tr-" + str(bool(self.train_logvar_t)) + '-'

        # Definition of the optimizer
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = torch.optim.Adadelta(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'asgd':
            optimizer = torch.optim.ASGD(self.network.parameters(),lr=learning_rate)
        learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, \
            step_size=learning_rate_steps,gamma=learning_rate_drop)

        # For all the epochs
        for epoch in range(n_epochs):

            print("Epoch #{}/{}".format(epoch,n_epochs))

            # Randomly sample a mini batch for the SGD
            for idx_sgd_batch, (sgd_train_x, sgd_train_y) in enumerate(progressbar(sgd_train_loader)):

                # Skip the last batch
                if idx_sgd_batch == n_sgd_batches - 1:
                    break

                # If we are not using the same batch for SGD and MI...
                if not same_batch:

                    # Randomly sample a mini batch for the MI estimation
                    idx_mi_batch, (mi_train_x, mi_train_y) = next(mi_train_batches)

                    # Prepare the MI loader again when finished
                    if (idx_mi_batch == n_mi_batches - 1):
                        mi_train_batches = enumerate(mi_train_loader)

                # If we use the same batch for SGD and MI...
                else:

                    mi_train_x, mi_train_y = sgd_train_x, sgd_train_y

                # Compute the best variance for the MoG (only 1 time per epoch)
                with torch.no_grad():
                    if idx_sgd_batch == 0:
                        mi_train_mean_t = self.network.encode(mi_train_x,random=False)
                        self.update_logvar_kde(mi_train_mean_t)

                # - Gradient descent
                optimizer.zero_grad()
                sgd_train_logits_y = self.network(sgd_train_x)
                sgd_train_ITY = self.get_ITY(sgd_train_logits_y,sgd_train_y)
                mi_train_mean_t = self.network.encode(mi_train_x,random=False)
                mi_train_IXT = self.get_IXT(mi_train_mean_t)
                loss = - 1.0 * (sgd_train_ITY - self.beta * mi_train_IXT)
                loss.backward()
                optimizer.step()

            # Update learning rate
            learning_rate_scheduler.step()

            # Report results
            if epoch % eval_rate == 0:
                with torch.no_grad():
                    epochs[report] = epoch

                    for _, (train_x, train_y) in enumerate(sgd_train_loader):
                        train_logits_y = self.network(train_x)
                        train_mean_t = self.network.encode(train_x,random=False)
                        train_IXT[report] += self.get_IXT(train_mean_t).item() / n_sgd_batches
                        train_ITY[report] += self.get_ITY(train_logits_y,train_y).item() / n_sgd_batches
                        train_loss[report] += - 1.0 * (train_ITY[report] - \
                            self.beta * train_IXT[report]) / n_sgd_batches
                        train_accuracy[report] += self.evaluate(train_logits_y,train_y).item() / n_sgd_batches

                    _, (validation_x, validation_y) = next(enumerate(validation_loader))
                    validation_logits_y = self.network(validation_x)
                    validation_mean_t = self.network.encode(validation_x,random=False)
                    validation_IXT[report] = self.get_IXT(validation_mean_t).item()
                    validation_ITY[report] = self.get_ITY(validation_logits_y,validation_y).item()
                    validation_loss[report] = - 1.0 * (validation_ITY[report] - \
                        self.beta * train_IXT[report])
                    validation_accuracy[report] = self.evaluate(validation_logits_y,validation_y).item()

                if verbose:
                    print("\n")
                    print("\n** Results report **")
                    print("- I(X;T) = " + str(train_IXT[report]))
                    print("- I(T;Y) = " + str(train_ITY[report]))
                    print("- Training accuracy: " + str(train_accuracy[report]))
                    print("- Validation accuracy: " + str(validation_accuracy[report]))
                    print("\n")

                report += 1

                # Visualize results
                if visualization:
                    with torch.no_grad():
                        if self.K == 2:
                            _, (visualize_x,visualize_y) = next(enumerate(validation_loader))
                            visualize_t = self.network.encode(visualize_x,random=True)
                        else:
                            visualize_y, visualize_t = None, None
                        plot_results(train_IXT[:report], validation_IXT[:report],
                            train_ITY[:report], validation_ITY[:report],
                            train_loss[:report], validation_loss[:report],
                            visualize_t, visualize_y, epochs[:report], self.HY, self.K,
                            fig, ax)

                # Save results
                if self.K == 2:
                    with torch.no_grad():
                        _, (visualize_x,visualize_y) = next(enumerate(validation_loader))
                        visualize_t = self.network.encode(visualize_x,random=True)
                    np.save(logs_dir + name_base + 'hidden_variables', visualize_t)
                np.save(logs_dir + name_base + 'train_IXT', train_IXT)
                np.save(logs_dir + name_base + 'validation_IXT', validation_IXT)
                np.save(logs_dir + name_base + 'train_ITY', train_ITY)
                np.save(logs_dir + name_base + 'validation_ITY', validation_ITY)
                np.save(logs_dir + name_base + 'train_loss', train_loss)
                np.save(logs_dir + name_base + 'validation_loss', validation_loss)
                np.save(logs_dir + name_base + 'train_accuracy', train_accuracy)
                np.save(logs_dir + name_base + 'validation_accuracy', validation_accuracy)
                np.save(logs_dir + name_base + 'epochs', epochs)
                plt.savefig(figs_dir + name_base + 'image.pdf', format = 'pdf')
                plt.savefig(figs_dir + name_base + 'image.png', format = 'png')
