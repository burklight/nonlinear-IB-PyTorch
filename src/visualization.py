import matplotlib.pyplot as plt
import numpy as np

def init_results_visualization(K):
    '''
    Initializes the results plot
    '''

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize = (16+1,4), gridspec_kw={'width_ratios': [1,1,1.25,1]})
    return fig, ax

def plot_results(IXT_train, IXT_validation, ITY_train, ITY_validation,
    loss_train, loss_validation, t, y, epochs, IXY, K, fig, ax, problem_type):
    '''
    Plots the results in figure fig
    '''

    IXY = IXY / np.log(2) # in bits

    # Print the IXT
    ax[0].clear()
    ax[0].plot(epochs, IXT_train, '-', color = 'red', label = 'train')
    ax[0].plot(epochs, IXT_validation, '-', color = 'blue', label = 'validation')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel(r'$I(X;T)$')
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(left=0)
    ax[0].legend()

    # Print the ITY
    lim = np.linspace(0,np.max(epochs),1000)
    ax[1].clear()
    ax[1].plot(lim, np.ones(1000)*IXY, color = 'black', linestyle = '--')
    ax[1].plot(epochs, ITY_train, '-', color = 'red', label = 'train')
    ax[1].plot(epochs, ITY_validation, '-', color = 'blue', label = 'validation')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel(r'$I(T;Y)$')
    ax[1].set_ylim(bottom=0, top=IXY*(1.1))
    ax[1].set_xlim(left=0)
    ax[1].legend()

    # Print the information plane
    maxval = max(IXY*1.1,max(np.max(IXT_train), np.max(IXT_validation)))
    diag = np.linspace(0,maxval,1000)
    ax[2].clear()
    ax[2].plot(diag, diag, color = 'darkorange', linestyle = '--')
    ax[2].plot(diag, np.ones(1000)*IXY, color = 'darkorange', linestyle = '--')
    ax[2].fill_between(diag, 0, np.where(diag>IXY, IXY, diag), alpha = 0.5, color='darkorange')
    ax[2].plot(diag, np.where(diag>IXY, IXY, diag), alpha = 0.5, color='blue', linewidth=4)
    ax[2].plot(IXT_train[:-1], ITY_train[:-1], 'X', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[2].plot(IXT_validation[:-1], ITY_validation[:-1], '.', color='blue', markersize=9, markeredgecolor='black', label = 'validation')
    ax[2].plot(IXT_train[-1], ITY_train[-1], '*', color='red', markersize=9, markeredgecolor='black')
    ax[2].plot(IXT_validation[-1], ITY_validation[-1], '*', color='blue', markersize=9, markeredgecolor='black')
    ax[2].plot(IXT_train, ITY_train, linestyle=':', color='red')
    ax[2].plot(IXT_validation, ITY_validation, linestyle=':', color='blue')
    ax[2].set_xlabel(r'$I(X;T)$')
    ax[2].set_ylabel(r'$I(T;Y)$')
    if problem_type == 'classification':
        ax[2].annotate(r' $I(X;Y) = H(Y)$', xy=(maxval*0.75,IXY*(1.05)), color='darkorange')
    else:
        ax[2].annotate(r' $I(X;Y)$', xy=(maxval*0.75,IXY*(1.05)), color='darkorange')
    ax[2].annotate(r' $I(X;T) \geq I(T;Y)$', xy=(IXY*(0.5),IXY*(1.05)), color='darkorange')
    ax[2].set_xlim(left=0,right=maxval)
    ax[2].set_ylim(bottom=0, top=IXY*(1.1))
    ax[2].legend()

    # Print the representations
    npoints = 10000
    ax[3].clear()
    if problem_type == 'classification':
        ax[3].scatter(t[0:npoints,0], t[0:npoints,1], c=y[0:npoints], cmap='tab10', marker='.', s=0.005)
    else:
        ax[3].scatter(t[0:npoints,0], t[0:npoints,1], c=y[0:npoints], cmap='seismic', marker='.', s=0.05)
        m, sd = np.median(t,0) , np.std(t,0).max()
        ax[3].set_xlim([m[0]-2.5*sd,m[0]+2.5*sd])
        ax[3].set_ylim([m[1]-2.5*sd,m[1]+2.5*sd])
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_xlabel('Bottleneck variable space')

    plt.tight_layout()

def plot_behavior(logs_dir,figs_dir,K,betas,hfunc_name,hfunc_param,IXY):

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (16,4))

    # Prepare the empty vectors
    IXT_train = np.empty(len(betas))
    IXT_validation = np.empty(len(betas))
    ITY_train = np.empty(len(betas))
    ITY_validation = np.empty(len(betas))

    # For all betas
    for i,beta in enumerate(betas):

        # Load the information plane point they obtained
        name_base = 'dimT-'+str(K) + '--beta-' + ('%.3f' % beta).replace('.',',') + '--hfunc-' \
            + hfunc_name + '--param-' + ('%.3f' % hfunc_param).replace('.',',') 
        IXT_train[i] = np.load(logs_dir + name_base + '--train_IXT.npy')[-1]
        IXT_validation[i] = np.load(logs_dir + name_base + '--validation_IXT.npy')[-1]
        ITY_train[i] = np.load(logs_dir + name_base + '--train_ITY.npy')[-1]
        ITY_validation[i] = np.load(logs_dir + name_base + '--validation_ITY.npy')[-1]

    # Print the information plane
    maxval = max(IXY*1.1,max(np.max(IXT_train), np.max(IXT_validation)))
    diag = np.linspace(0,maxval,1000)
    ax[0].plot(diag, diag, color = 'darkorange', linestyle = '--')
    ax[0].plot(diag, np.ones(1000)*IXY, color = 'darkorange', linestyle = '--')
    ax[0].fill_between(diag, 0, np.where(diag>IXY, IXY, diag), alpha = 0.5, color='darkorange')
    ax[0].plot(diag, np.where(diag>IXY, IXY, diag), alpha = 0.5, color='blue', linewidth=4)
    ax[0].plot(IXT_train, ITY_train, 'X:', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[0].plot(IXT_validation, ITY_validation, '.:', color='blue', markersize=9, markeredgecolor='black', label = 'validation')
    ax[0].set_xlabel(r'$I(X;T)$')
    ax[0].set_ylabel(r'$I(T;Y)$')
    ax[0].annotate(r' $I(X;Y) = H(Y)$', xy=(maxval*0.75,IXY*(1.05)), color='darkorange')
    ax[0].annotate(r' $I(X;T) \geq I(T;Y)$', xy=(IXY*(1.05),IXY*(1.05)), color='darkorange')
    ax[0].set_xlim(left=0,right=maxval)
    ax[0].set_ylim(bottom=0, top=IXY*(1.1))
    ax[0].legend()

    # Print the evolution of I(T;Y)
    diag = np.linspace(np.min(betas),np.max(betas),1000)
    ax[1].plot(diag, np.ones(1000)*IXY, color = 'darkorange', linestyle = '--')
    ax[1].plot(betas, ITY_train, 'X:', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[1].plot(betas, ITY_validation, '.:', color = 'blue', markersize=9, markeredgecolor='black', label='validation')
    ax[1].set_xlabel(r'$\beta$')
    ax[1].set_ylabel(r'$I(X;T)$')
    ax[1].annotate(r' $I(X;Y) = H(Y)$', xy=(betas[-1]*0.75,IXY*(1.05)), color='darkorange')
    ax[1].set_xlim(left=0,right=np.max(betas))
    ax[1].set_ylim(bottom=0,top=IXY*(1.1))
    ax[1].legend()

    # Print the evolution of I(X;T)
    diag = np.linspace(np.min(betas),np.max(betas),1000)
    ax[2].plot(diag, np.ones(1000)*IXY, color = 'darkorange', linestyle = '--')
    ax[2].plot(betas, IXT_train, 'X:', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[2].plot(betas, IXT_validation, '.:', color = 'blue', markersize=9, markeredgecolor='black', label='validation')
    ax[2].set_xlabel(r'$\beta$')
    ax[2].set_ylabel(r'$I(X;T)$')
    ax[2].annotate(r' $I(X;Y) = H(Y)$', xy=(betas[-1]*0.75,IXY*(1.05)), color='darkorange')
    ax[2].set_xlim(left=0,right=np.max(betas))
    ax[2].set_ylim(bottom=0,top=maxval)
    ax[2].legend()

    plt.tight_layout()
    plt.plot()

    name_base = 'dimT-'+str(K) + '--NB-' + str(len(betas)) + '--hfunc-' \
            + hfunc_name + '--param-' + ('%.3f' % hfunc_param).replace('.',',') 
    plt.savefig(figs_dir + name_base + 'behavior.pdf', format = 'pdf')
    plt.savefig(figs_dir + name_base + 'behavior.png', format = 'png')
