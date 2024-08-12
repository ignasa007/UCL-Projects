import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt
from dataset import train_S, test_S, S

FONTDICT = {'family':'serif', 'size':12, 'weight':'normal'}

def expand_dims(*arrays):

    for array in arrays:
        
        if not isinstance(array, (tuple, list, np.ndarray)):
            array = list(array)
        array = np.array(array)
        
        if array.ndim == 1:
            array = array[None, :]
        assert isinstance(array, np.ndarray) and array.ndim == 2
        
        yield array

def evaluate(coreset_k, alpha_k, svm_kernel):

    def metrics(true, preds):
        accuracy = accuracy_score(true, preds)
        f1 = f1_score(true, preds)
        bce = log_loss(true, preds)
        return accuracy, f1, bce

    train_preds = (train_S[coreset_k, -1] * alpha_k[coreset_k]) @ (svm_kernel(train_S[coreset_k, :-1], train_S[:, :-1]) + 1)
    train_preds = np.where(train_preds < 0., -1., +1.)
    train_metrics = metrics(train_S[:, -1], train_preds)

    test_preds = (train_S[coreset_k, -1] * alpha_k[coreset_k]) @ (svm_kernel(train_S[coreset_k, :-1], test_S[:, :-1]) + 1)
    test_preds = np.where(test_preds < 0., -1., +1.)
    test_metrics = metrics(test_S[:, -1], test_preds)

    return train_metrics, test_metrics
    
def plot_decision(coreset_k, alpha_k, svm_kernel):

    x = np.arange(np.min(S[:, 0])-0.1, np.max(S[:, 0])+0.1, 0.01)
    y = np.arange(np.min(S[:, 1])-0.1, np.max(S[:, 1])+0.1, 0.01)
    xv, yv = np.meshgrid(x, y)
    test_inputs = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
    predictions = (train_S[coreset_k, -1] * alpha_k[coreset_k]) @ (svm_kernel(train_S[coreset_k, :-1], test_inputs) + 1)
    predictions = np.where(predictions < 0., -1., +1.)

    plt.scatter(test_inputs[:, 0][predictions == +1.], test_inputs[:, 1][predictions == +1.], s=12, color='lightgreen')
    plt.scatter(test_inputs[:, 0][predictions == -1.], test_inputs[:, 1][predictions == -1.], s=12, color='lightblue')
    plt.scatter(S[:, 0][S[:, -1] == +1.], S[:, 1][S[:, -1] == +1.], s=20, color='darkgreen')
    plt.scatter(S[:, 0][S[:, -1] == -1.], S[:, 1][S[:, -1] == -1.], s=20, color='darkblue')

    plt.tight_layout()
    plt.grid()
    plt.savefig('assets/2-circles-decision-boundary.png')
    plt.show()

def plot_convergence(dual_iterates, dual_evaluations):

    opt_eval = dual_evaluations[-1]
    gap_eval = opt_eval-np.array(dual_evaluations)[:-1]
    plt.plot(gap_eval, color='blue')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel(r'$f(\alpha^\ast)-f(\alpha_k)$', fontdict=FONTDICT)
    plt.grid()
    plt.tight_layout()
    plt.savefig('assets/gap_eval.png')
    plt.show()

    quotient_eval = gap_eval[1:] / gap_eval[:-1]
    plt.plot(quotient_eval, color='blue')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel(r'$\frac{f(\alpha^\ast)-f(\alpha_{k+1})}{f(\alpha^\ast)-f(\alpha_{k})}$', fontdict=FONTDICT)
    plt.grid()
    plt.tight_layout()
    plt.savefig('assets/quotient_eval.png')
    plt.show()

    opt_iterate = dual_iterates[-1]
    gap_iterate = np.linalg.norm(opt_iterate-np.array(dual_iterates)[:-1], 2, axis=1)
    plt.plot(gap_iterate, color='blue')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel(r'$\|\alpha^\ast-\alpha_k\|_2$', fontdict=FONTDICT)
    plt.grid()
    plt.tight_layout()
    plt.savefig('assets/gap_iterate.png')
    plt.show()
    
    quotient_iterate = gap_iterate[1:] / gap_iterate[:-1]
    plt.plot(quotient_iterate, color='blue')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel(r'$\frac{\|\alpha^\ast-\alpha_{k+1}\|_2}{\|\alpha^\ast-\alpha_k\|_2}$', fontdict=FONTDICT)
    plt.grid()
    plt.tight_layout()
    plt.savefig('assets/quotient_iterate.png')
    plt.show()

def plot_coreset(coreset_sizes):

    plt.plot(coreset_sizes, color='blue')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel(r'$|\mathcal{I}_k|$', fontdict=FONTDICT)
    plt.grid()
    plt.tight_layout()
    plt.savefig('assets/coreset_sizes.png')
    plt.show()

def plot_steps(step_sizes):

    plt.plot(step_sizes, color='blue')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel(r'$\lambda_k$', fontdict=FONTDICT)
    plt.grid()
    plt.tight_layout()
    plt.savefig('assets/step_sizes.png')
    plt.show()

def plot_metrics(train_metrics, test_metrics):

    train_acc, train_f1, train_loss = zip(*train_metrics)
    test_acc, test_f1, test_loss = zip(*test_metrics)

    plt.plot(train_acc, color='blue', label='Train')
    plt.plot(test_acc, color='green', label='Test')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel('Accuracy', fontdict=FONTDICT)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/accuracy.png')
    plt.show()

    plt.plot(train_f1, color='blue', label='Train')
    plt.plot(test_f1, color='green', label='Test')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel('F1-Score', fontdict=FONTDICT)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/f1_score.png')
    plt.show()

    plt.plot(train_loss, color='blue', label='Train')
    plt.plot(test_loss, color='green', label='Test')
    plt.xlabel('Iteration', fontdict=FONTDICT)
    plt.ylabel('BCE', fontdict=FONTDICT)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/bce_loss.png')
    plt.show()