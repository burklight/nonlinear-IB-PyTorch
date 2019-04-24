import matplotlib.pyplot as plt
import numpy as np

def init_results_visualization():
    '''
    Initializes the results plot
    '''

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (20,4))
    return fig, ax

def plot_results(IXT_train, IXT_validation, ITY_train, ITY_validation,
    loss_train, loss_validation, t, y, epochs, HY, fig, ax):
    '''
    Plots the results in figure fig
    '''

    HY = HY / np.log(2) # in bits

    # Print the Loss
    ax[0].clear()
    ax[0].plot(epochs, loss_train, '-', color = 'red', label = 'train')
    ax[0].plot(epochs, loss_validation, '-', color = 'blue', label = 'validation')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel(r'$-\mathcal{L}_{IB}(T)$')
    ax[0].set_xlim(left=0)
    ax[0].legend()

    # Print the IXT
    ax[1].clear()
    ax[1].plot(epochs, IXT_train, '-', color = 'red', label = 'train')
    ax[1].plot(epochs, IXT_validation, '-', color = 'blue', label = 'validation')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel(r'$I(X;T)$')
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=0)
    ax[1].legend()

    # Print the ITY
    lim = np.linspace(0,np.max(epochs),1000)
    ax[2].clear()
    ax[2].plot(lim, np.ones(1000)*HY, color = 'black', linestyle = '--')
    ax[2].plot(epochs, ITY_train, '-', color = 'red', label = 'train')
    ax[2].plot(epochs, ITY_validation, '-', color = 'blue', label = 'validation')
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel(r'$I(T;Y)$')
    ax[2].set_ylim(bottom=0, top=HY*(1.1))
    ax[2].set_xlim(left=0)
    ax[2].legend()

    # Print the information plane
    maxval = max(HY*1.1,max(np.max(IXT_train), np.max(IXT_validation)))
    diag = np.linspace(0,maxval,1000)
    ax[3].clear()
    ax[3].plot(diag, diag, color = 'black', linestyle = '--')
    ax[3].plot(diag, np.ones(1000)*HY, color = 'black', linestyle = '--')
    ax[3].plot(IXT_train[:-1], ITY_train[:-1], 'X', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[3].plot(IXT_validation[:-1], ITY_validation[:-1], 'X', color='blue', markersize=9, markeredgecolor='black', label = 'validation')
    ax[3].plot(IXT_train[-1], ITY_train[-1], '*', color='red', markersize=9, markeredgecolor='black')
    ax[3].plot(IXT_validation[-1], ITY_validation[-1], '*', color='blue', markersize=9, markeredgecolor='black')
    ax[3].plot(IXT_train, ITY_train, linestyle=':', color='red')
    ax[3].plot(IXT_validation, ITY_validation, linestyle=':', color='blue')
    ax[3].set_xlabel(r'$I(X;T)$')
    ax[3].set_ylabel(r'$I(T;Y)$')
    ax[3].set_xlim(left=0,right=maxval)
    ax[3].set_ylim(bottom=0, top=HY*(1.1))
    ax[3].legend()

    # Print the representations
    npoints = 10000
    ax[4].clear()
    ax[4].scatter(t[0:npoints,0], t[0:npoints,1], c=y[0:npoints], cmap='tab10', marker='.', alpha=0.1)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].set_xlabel('Bottleneck variable space')

    plt.tight_layout()
    plt.pause(0.01)
