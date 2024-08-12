import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fontdict = {'font':'serif', 'size':12}

def error_convergence(errors, xlog=True, ylog=True, overlay_fit=True, ignore=0):

    y_fit = errors[errors != 0.] if ylog else errors
    x_fit = np.arange(y_fit.size)
    x_fit = x_fit if not xlog else x_fit+1

    plt.figure(figsize=(6, 4))
    plt.plot(x_fit, y_fit, color='blue', label='Empirical errors')
    plt.xlabel(r'$k$', fontdict=fontdict)
    plt.ylabel(r'error($x_k$) = $\|x^*-x_k\|_2$', fontdict=fontdict)
    
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    
    if overlay_fit:
        x_fit = x_fit[ignore:]; y_fit = y_fit[ignore:]
        x_fit = x_fit.reshape(-1, 1)
        lr = LinearRegression().fit(
            x_fit if not xlog else np.log10(x_fit),
            y_fit if not ylog else np.log10(y_fit)
        )
        preds = lr.predict(x_fit if not xlog else np.log10(x_fit))
        plt.plot(x_fit.flatten(), preds if not ylog else np.power(10, preds), '--', color='green', label=fr'Linear fit, $m = {round(lr.coef_[0], 3)}$')
        plt.legend()
    
    plt.grid()
    plt.tight_layout()
    plt.show()

def quotient_convergence(errors, order=1, xlog=False, ylog=False):

    errors = errors[errors != 0.]
    quotients = errors[1:] / np.power(errors[:-1], order)

    plt.figure(figsize=(6, 4))
    plt.plot(quotients, color='blue')
    plt.xlabel(r'$k$', fontdict=fontdict)
    if order == 1:
        plt.ylabel(fr'$\|x^*-x_{{k+1}}\|_2\ /\ \|x^*-x_{{k}}\|_2$', fontdict=fontdict)
    else:
        plt.ylabel(fr'$\|x^*-x_{{k+1}}\|_2\ /\ \|x^*-x_{{k}}\|_2^{order}$', fontdict=fontdict)
    
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    
    plt.grid()
    plt.tight_layout()
    plt.show()

def gradient_norm_convergence(xs, gradient, xlog=False, ylog=False, overlay_fit=False, ignore=0):

    y_fit = np.array([np.linalg.norm(gradient(x)) for x in xs])
    x_fit = np.arange(y_fit.size)
    x_fit = x_fit if not xlog else x_fit+1

    plt.figure(figsize=(6, 4))
    plt.plot(x_fit, y_fit, color='blue', label='Gradient norms')
    plt.xlabel(r'$k$', fontdict=fontdict)
    plt.ylabel(r'$\|\nabla f(x_k)\|_2$', fontdict=fontdict)
    if ylog:
        plt.yscale('log')

    if overlay_fit:
        x_fit = x_fit[ignore:]; y_fit = y_fit[ignore:]
        x_fit = x_fit.reshape(-1, 1)
        lr = LinearRegression().fit(
            x_fit if not xlog else np.log10(x_fit),
            y_fit if not ylog else np.log10(y_fit)
        )
        preds = lr.predict(x_fit if not xlog else np.log10(x_fit))
        plt.plot(x_fit.flatten(), preds if not ylog else np.power(10, preds), '--', color='green', label=fr'Linear fit, $m = {round(lr.coef_[0], 3)}$')
        plt.legend()

    plt.grid()
    plt.tight_layout()
    plt.show()

def trajectories_convergence(xs, f, start_idx, middle_idx, end_idx, xlims, ylims, arrows=True):

    cols = 2 if end_idx is not None else 1
    plt.figure(figsize=(6*cols, 4))
    fontdict = {'font':'serif', 'size': 11}
    
    plt.subplot(1, cols, 1)
    x, y = np.meshgrid(np.linspace(*(xlims[0]), 1000), np.linspace(*(ylims[0]), 1000))
    z = f(x, y)
    cs = plt.contourf(x, y, z, levels=100, cmap='RdYlBu')
    cbar = plt.colorbar(cs)
    cbar.formatter.set_powerlimits((0, 0))
    plt.xticks(rotation=45)

    if arrows:
        for x_old, x_new in zip(xs[start_idx:middle_idx], xs[start_idx+1:middle_idx+1]):
            plt.annotate(
                '', xy=x_new.flatten(), xytext=x_old.flatten(),
                arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1},
                va='center', ha='center'
            )
    else:
        iterates = np.array(xs).squeeze()
        plt.plot(
            iterates[start_idx:middle_idx+1,0], 
            iterates[start_idx:middle_idx+1,1], 
            marker='o', markersize=3.5, linestyle='-', color='black'
        )
    
    start, middle = tuple([*np.round(xs[start_idx], 3).flatten()]), tuple([*np.round(xs[middle_idx], 3).flatten()])
    plt.title(
        'Optimization trajectory from' 
        '\n' 
        rf'$x_{{{start_idx}}} = {start}$  to  $x_{{{middle_idx}}} = {middle}$', 
        fontdict=fontdict
    )

    if end_idx is not None:
        
        plt.subplot(1, 2, 2)
        x, y = np.meshgrid(np.linspace(*(xlims[1]), 1000), np.linspace(*(ylims[1]), 1000))
        z = f(x, y)
        cs = plt.contourf(x, y, z, levels=100, cmap='RdYlBu')
        cbar = plt.colorbar(cs)
        cbar.formatter.set_powerlimits((0, 0))
        plt.xticks(rotation=45)

        if arrows:
            for x_old, x_new in zip(xs[middle_idx:end_idx], xs[middle_idx+1:end_idx+1]):
                plt.annotate(
                    '', xy=x_new.flatten(), xytext=x_old.flatten(),
                    arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1},
                    va='center', ha='center'
                )
        else:
            iterates = np.array(xs).squeeze()
            plt.plot(
                iterates[middle_idx:end_idx+1,0], 
                iterates[middle_idx:end_idx+1,1], 
                marker='o', markersize=3.5, linestyle='-', color='black'
            )
        
        middle, end = tuple([*np.round(xs[middle_idx], 3).flatten()]), tuple([*np.round(xs[end_idx], 3).flatten()])
        plt.title(
            'Optimization trajectory from' 
            '\n' 
            rf'$x_{{{middle_idx}}} = {middle}$  to  $x_{{{end_idx}}} = {end}$', 
            fontdict=fontdict
        )

    plt.tight_layout()
    plt.show()