# TODO: test with c2 = 0.9 for steepest descent and newton methods
def line_search_wolfe(f, gradient, x_k, p_k, alpha_max, c1=1e-4, c2=1e-1, w=0.9, max_iter=10):
    
    def phi(alpha):
        x = x_k + alpha*p_k
        return f(*x)

    def dphi(alpha):
        x = x_k + alpha*p_k
        return float('inf') if f(*x) == float('inf') else (gradient(x).T @ p_k).squeeze().item()
    
    def zoom_in(alpha_l, alpha_h):
        alpha = alpha_l; stop = False
        for _ in range(max_iter):
            alpha_j = (alpha_l+alpha_h)/2
            phi_j = phi(alpha_j)
            if alpha_h - alpha_l < 1e-8:
                alpha = alpha_j; stop = True
            if (phi_j > phi(0) + c1*alpha_j*dphi(0)) or (phi_j >= phi(alpha_l)):
                alpha_h = alpha_j
            else:
                dphi_j = dphi(alpha_j)
                if abs(dphi_j) <= -c2*dphi(0):
                    alpha = alpha_j; stop = True
                elif dphi_j*(alpha_h-alpha_l) >= 0:
                    alpha_h = alpha_l
                alpha_l = alpha_j
            if stop:
                break
        return alpha
    
    alphas = [0, alpha_max]
    phi_i = [phi(0)]; dphi_i = [dphi(0)]
    n = 1; stop = False

    while not stop:
        
        phi_i.append(phi(alphas[n]))
        dphi_i.append(dphi(alphas[n]))
        
        # if the first Wolfe condition fails, i.e. we took a very big step and f increased => we want a
        # smaller step size => zoom in going up from the previously accepted size to the one that failed
        if (phi_i[n] > phi_i[0] + c1*alphas[n]*dphi_i[0]) or (n > 1 and phi_i[n] >= phi_i[n-1]):
            alpha_s = zoom_in(alphas[n-1], alphas[n]); stop = True
        
        # if the first Wolfe condition passes and the second condition passes too
        elif abs(dphi_i[n]) <= -c2*dphi_i[0]:
            alpha_s = alphas[n]; stop = True
        
        # second Wolfe condition failed, but our step size is still too big -- f increases from the new point
        # in the direction of p_k => we zoom in going down from this step size to the previously accepted one
        elif dphi_i[n] >= 0:
            alpha_s = zoom_in(alphas[n], alphas[n-1]); stop = True

        # choose alphas[n+1] in (alpha[n], alpha_max)
        alphas.append(w*alphas[n] + (1-w)*alpha_max)
        if n == max_iter:
            alpha_s = alphas[-1]
            break

        n += 1

    return alpha_s