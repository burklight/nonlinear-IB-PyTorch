import torch
import math
from progressbar import progressbar
import numpy as np
from network import nlIB_network
from kde_estimation_mi import KDE_IXT_estimation
from visualization import plot_results
from visualization import init_results_visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class NonlinearIB(torch.nn.Module):
    '''
    Implementation of the Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    '''

    def __init__(self,n_x,n_y,problem_type,K,beta,hfunc='exp',param=1,logvar_t=-1.0,train_logvar_t=False):
        super(NonlinearIB,self).__init__()

        self.HY = np.log(n_y) # in natts
        self.varY = 0 # to be updated with the training dataset

        self.K = K
        self.beta = beta

        if hfunc == 'exp':
            self.hfunc = lambda r : torch.exp(param*r)
        elif hfunc == 'pow':
            self.hfunc = lambda r : r ** (1.0+param)
        else:
            self.hfunc = lambda r : r
        self.hfunc_name = hfunc
        self.hfunc_param = param

        self.train_logvar_t = train_logvar_t
        self.network = nlIB_network(K,n_x,n_y,logvar_t,self.train_logvar_t)

        self.problem_type = problem_type
        if problem_type == 'classification':
            self.ce = torch.nn.CrossEntropyLoss()
        else:
            self.mse = torch.nn.MSELoss()

    def get_IXT(self,mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.network.logvar_t,mean_t) # in natts
        self.IXT = IXT / np.log(2) # in bits
        return self.IXT

    def get_ITY(self,logits_y,y):
        '''
        Obtains the mutual information between the bottleneck variable and the output.
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        if self.problem_type == 'classification':
            HY_given_T = self.ce(logits_y,y)
            self.ITY = (self.HY - HY_given_T) / np.log(2) # in bits
        else:
            MSE = self.mse(logits_y,y)
            self.ITY = 0.5 * torch.log(self.varY / MSE) / np.log(2) # in bits
        return self.ITY
    
    def get_loss(self,IXT,ITY):
        '''
        Returns the loss function from the XXXX et al. 2019, "The Convex Information Bottleneck Lagrangian"
        Paramters: 
        - IXT (float) : Mutual information between X and T
        - ITY (float) : Mutual information between T and Y 
        '''
        
        loss = -1.0 * (ITY - self.beta * self.hfunc(IXT))
        return loss

    def evaluate(self,logits_y,y):
        '''
        Evauluates the performance of the model (accuracy) for classification or (mse) for regression
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        with torch.no_grad():
            if self.problem_type == 'classification':
                y_hat = y.eq(torch.max(logits_y,dim=1)[1])
                accuracy = torch.mean(y_hat.float())
                return accuracy
            else:
                mse = self.mse(logits_y,y) # logits y is y_hat in regression
                return mse
    
    def _evaluate_network(self,x,y,n_batches,IXT=0,ITY=0,loss=0,performance=0):

        logits_y = self.network(x)
        mean_t = self.network.encode(x,random=False)
        IXT += self.get_IXT(mean_t) / n_batches
        ITY += self.get_ITY(logits_y,y) / n_batches
        loss += self.get_loss(IXT,ITY) / n_batches
        performance += self.evaluate(logits_y,y) / n_batches

        return IXT, ITY, loss, performance

    def evaluate_network(self,dataset_loader,n_batches):

        IXT = 0
        ITY = 0
        loss = 0
        performance = 0
        if n_batches > 1:
            for _,(x,y) in enumerate(dataset_loader):
                IXT, ITY, loss, performance = self._evaluate_network(x,y,n_batches,IXT,ITY,loss,performance)
        else:
            _, (x,y) = next(enumerate(dataset_loader))
            IXT, ITY, loss, performance = self._evaluate_network(x,y,n_batches)
        
        return IXT.item(), ITY.item(), loss.item(), performance.item()


    def fit(self,trainset,validationset,testset,n_epochs=200,optimizer_name='adam',learning_rate=0.0001,\
        learning_rate_drop=0.6,learning_rate_steps=10, sgd_batch_size=128, eval_rate=20, early_stopping_lim=15,
        verbose=True,visualization=True, logs_dir='.',figs_dir='.'):
        '''
        Trains the model with the training set with early stopping with the validation set. Evaluates on the test set.
        Parameters:
        - trainset (PyTorch Dataset) : Training dataset
        - validationset (PyTorch Dataset) : Validation dataset
        - testset (PyTorch Dataset) : Test dataset
        - n_epochs (int) : number of training epochs
        - optimizer_name (str) : name of the optimizer 
        - learning_rate (float) : initial learning rate
        - learning_rate_drop (float) : multicative learning decay factor
        - learning_rate_steps (int) : number of steps before decaying the learning rate
        - sgd_batch_size (int) : size of the SGD mini-batch
        - eval_rate (int) : the model is evaluated every eval_rate epochs
        - early_stopping_lim (int) : number of epochs for the validation set not to increase so that the training stops
        - verbose (bool) : if True, the evaluation is reported
        - visualization (bool) : if True, the evaluation is shown
        - logs_dir (str) : path for the storage of the evaluation
        - figs_dir (str) : path for the storage of the images of the evaluation
        '''

        # Definition of the training and test losses, accuracies and MI
        report = 0
        n_reports = math.ceil(n_epochs / eval_rate)
        train_loss_vec = np.zeros(n_reports)
        test_loss_vec = np.zeros(n_reports)
        train_performance_vec = np.zeros(n_reports)
        test_performance_vec = np.zeros(n_reports)
        train_IXT_vec = np.zeros(n_reports)
        train_ITY_vec = np.zeros(n_reports)
        test_IXT_vec = np.zeros(n_reports)
        test_ITY_vec = np.zeros(n_reports)
        epochs_vec = np.zeros(n_reports)
        early_stopping_count = 0
        loss_prev = math.inf 

        # If regression we update the variance of the output 
        if self.problem_type == 'regression':
            self.varY = torch.var(trainset.targets)
            self.HY = 0.5 * math.log(self.varY.item()*2.0*math.pi*math.e) # in natts

        # Data Loader
        n_sgd_batches = math.floor(len(trainset) / sgd_batch_size)
        train_loader = torch.utils.data.DataLoader(trainset, \
            batch_size=sgd_batch_size,shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validationset, \
            batch_size=len(validationset),shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, \
            batch_size=len(testset),shuffle=False)

        # Prepare visualization
        if visualization:
            fig, ax = init_results_visualization(self.K)
            n_points = 10000
            pca = PCA(n_components=2)

        # Prepare name for figures and logs
        name_base = 'dimT-'+str(self.K) + '--beta-' + ('%.3f' % self.beta).replace('.',',') + '--hfunc-' \
            + self.hfunc_name + '--param-' + ('%.3f' % self.hfunc_param).replace('.',',') 

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
            for train_x, train_y in progressbar(train_loader):

                # - Gradient descent
                optimizer.zero_grad()
                train_logits_y = self.network(train_x)
                train_ITY = self.get_ITY(train_logits_y,train_y)
                train_mean_t = self.network.encode(train_x,random=False)
                train_IXT = self.get_IXT(train_mean_t)
                loss = self.get_loss(train_IXT,train_ITY)
                loss.backward()
                optimizer.step()

            # Update learning rate
            #learning_rate_scheduler.step()

            # Report results
            if epoch % eval_rate == 0:
                with torch.no_grad():
                    epochs_vec[report] = epoch

                    train_IXT_vec[report], train_ITY_vec[report], train_loss_vec[report], train_performance_vec[report] = \
                        self.evaluate_network(train_loader,n_sgd_batches)
                    test_IXT_vec[report], test_ITY_vec[report], test_loss_vec[report], test_performance_vec[report] = \
                        self.evaluate_network(test_loader,1)

                if verbose:
                    print('\n\n** Results report **')
                    print(f'- Train | Test I(X;T) = {train_IXT_vec[report]} | {test_IXT_vec[report]}')
                    print(f'- Train | Test I(T;Y) = {train_ITY_vec[report]} | {test_ITY_vec[report]}')
                    if self.problem_type == 'classification':
                        print(f'- Train | Test accuracy = {train_performance_vec[report]} | {test_performance_vec[report]}\n')
                    else:
                        print(f'- Train | Test MSE = {train_performance_vec[report]} | {test_performance_vec[report]}\n')

                report += 1

                # Visualize results and save results
                if visualization:
                    with torch.no_grad():
                        _, (visualize_x,visualize_y) = next(enumerate(test_loader))
                        visualize_t = self.network.encode(visualize_x[:n_points],random=False)
                        
                        plot_results(train_IXT_vec[:report], test_IXT_vec[:report],
                            train_ITY_vec[:report], test_ITY_vec[:report],
                            train_loss_vec[:report], test_loss_vec[:report],
                            pca.fit_transform(visualize_t), visualize_y[:n_points], epochs_vec[:report], self.HY, self.K,
                            fig, ax, self.problem_type)
                        
                        plt.savefig(figs_dir + name_base + '--image.pdf', format = 'pdf')
                        plt.savefig(figs_dir + name_base + '--image.png', format = 'png')

                        print('The image is updated at ' + figs_dir + name_base + '--image.png')

                        np.save(logs_dir + name_base + '--hidden_variables', visualize_t)
                        np.save(logs_dir + name_base + '--labels', visualize_y)
                
                # Save the other results
                with torch.no_grad():
                    np.save(logs_dir + name_base + '--train_IXT', train_IXT_vec[:report])
                    np.save(logs_dir + name_base + '--validation_IXT', test_IXT_vec[:report])
                    np.save(logs_dir + name_base + '--train_ITY', train_ITY_vec[:report])
                    np.save(logs_dir + name_base + '--validation_ITY', test_ITY_vec[:report])
                    np.save(logs_dir + name_base + '--train_loss', train_loss_vec[:report])
                    np.save(logs_dir + name_base + '--validation_loss', test_loss_vec[:report])
                    np.save(logs_dir + name_base + '--train_performance', train_performance_vec[:report])
                    np.save(logs_dir + name_base + '--validation_performance', test_performance_vec[:report])
                    np.save(logs_dir + name_base + '--epochs', epochs_vec)
                
            # Check for early stopping 
            with torch.no_grad():
                _,_,loss_curr,_ = self.evaluate_network(validation_loader,1)
                if loss_curr >= loss_prev:
                    early_stopping_count += 1
                else:
                    early_stopping_count = 0
                if early_stopping_count >= early_stopping_lim:
                    break


