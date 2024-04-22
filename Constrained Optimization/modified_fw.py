import numpy as np
from utils import expand_dims, evaluate
from dataset import train_S as S

class ModifiedFW:

    def __init__(self, meb_kernel):

        self.meb_kernel = meb_kernel

    def initialize(self):

        pos_indices, = np.where(S[:, -1] == +1.)
        neg_indices, = np.where(S[:, -1] == -1.)
        distance_matrix = self.meb_kernel.compute_svm_kernel(pos_indices, neg_indices)
        assert distance_matrix.shape == (len(pos_indices), len(neg_indices))
        
        _j, _k = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        j, k = int(pos_indices[_j]), int(neg_indices[_k])
        coreset = [j, k]; self.meb_kernel.init_coreset(coreset)
        alpha = np.zeros(len(S)); alpha[j] = alpha[k] = 0.5
        assert alpha.shape == (len(S),) and np.isclose(np.sum(alpha), 1.)

        return coreset, alpha

    def optimize(self, eps):

        def compute_R():
            
            non_zero_duals, = expand_dims(alpha_k[coreset_k])
            assert non_zero_duals.shape == (1, len(coreset_k))
            R = (non_zero_duals @ self.meb_kernel.coreset_precomputations[coreset_k] @ non_zero_duals.T).squeeze()
            assert R > 0.
            
            return R
        
        def search_directions():
            
            non_zero_duals, = expand_dims(alpha_k[coreset_k])
            assert non_zero_duals.shape == (1, len(coreset_k))
            
            gamma_sq = (self.meb_kernel.coreset_precomputations @ non_zero_duals.T).squeeze()
            assert gamma_sq.shape == (len(S),)
            
            i_star, j_star = np.argmin(gamma_sq), coreset_k[np.argmax(gamma_sq[coreset_k])]
            assert j_star in coreset_k
            
            gamma_sq_i_star, gamma_sq_j_star = delta_sq+R_k-2*gamma_sq[i_star], delta_sq+R_k-2*gamma_sq[j_star]
            assert gamma_sq_i_star >= 0. and gamma_sq_j_star >= 0.
            
            delta_plus, delta_minus = gamma_sq_i_star/r_k_sq - 1., 1. - gamma_sq_j_star/r_k_sq
            assert delta_plus > 0. or delta_minus > 0.
            
            return i_star, delta_plus, j_star, delta_minus

        k = 0
        coreset_k, alpha_k = self.initialize()
        delta_sq = self.meb_kernel.radius_sq
        R_k = compute_R()
        r_k_sq = delta_sq - R_k
        assert r_k_sq > 0.
        i_star, delta_plus, j_star, delta_minus = search_directions()

        train_metrics, test_metrics = map(lambda x: [x], evaluate(coreset_k, alpha_k, self.meb_kernel.svm_kernel))
        dual_iterates, dual_evaluations, coreset_sizes = [alpha_k], [r_k_sq], [2]
        step_sizes = list()

        while 1+delta_plus > (1+eps)**2:

            coreset_sizes.append(coreset_sizes[-1])
            
            if delta_plus >= delta_minus:
            
                if i_star not in coreset_k:
                    coreset_k.append(i_star)
                    self.meb_kernel.add_precomputations(i_star)
                    coreset_sizes[-1] += 1
            
                lambda_k = delta_plus/(2*(1+delta_plus)); step_sizes.append(lambda_k)
                k += 1
                e_star = np.zeros_like(alpha_k); e_star[i_star] = 1.
                alpha_k = (1-lambda_k)*alpha_k + lambda_k*e_star
                r_k_sq = r_k_sq * (1 + (delta_plus**2)/(4*(1+delta_plus)))
            
            else:
            
                best_step = lambda_k = delta_minus / (2*(1-delta_minus))
                max_feasible = alpha_k[j_star] / (1-alpha_k[j_star])
            
                if lambda_k > max_feasible:
                    lambda_k = max_feasible
                    to_remove = coreset_k.index(j_star); coreset_k.pop(to_remove)
                    self.meb_kernel.remove_precomputations(to_remove)
                    coreset_sizes[-1] -= 1
                
                step_sizes.append(-lambda_k)
                k += 1
                e_star = np.zeros_like(alpha_k); e_star[j_star] = 1.
                alpha_k = (1+lambda_k)*alpha_k - lambda_k*e_star
                r_k_sq = (1+lambda_k) * r_k_sq * (1-lambda_k*(delta_minus-1))

            dual_iterates.append(alpha_k)
            dual_evaluations.append(r_k_sq)
            for logs, metrics in zip((train_metrics, test_metrics), evaluate(coreset_k, alpha_k, self.meb_kernel.svm_kernel)):
                logs.append(metrics)
            
            assert alpha_k.shape == (len(S),) and np.all(alpha_k>=0.) and np.isclose(np.sum(alpha_k), 1.)
            assert r_k_sq > 0.
            
            R_k = compute_R()
            i_star, delta_plus, j_star, delta_minus = search_directions()
            
        return dual_iterates, dual_evaluations, coreset_sizes, coreset_k, step_sizes, train_metrics, test_metrics