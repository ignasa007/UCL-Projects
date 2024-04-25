from svm_kernels import GaussianKernel
from meb_kernel import MEBKernel
from modified_fw import ModifiedFW
from utils import plot_decision, plot_convergence, plot_coreset, plot_steps, plot_metrics

svm_kernel = GaussianKernel(gamma=7e-1)
meb_kernel = MEBKernel(svm_kernel, C=1e3)
mfw_meb = ModifiedFW(meb_kernel)

dual_iterates, dual_evaluations, coreset_sizes, coreset_k, step_sizes, train_metrics, test_metrics = mfw_meb.optimize(eps=1e-6)
# plot_decision(coreset_k, dual_iterates[-1], svm_kernel)
# plot_convergence(dual_iterates, dual_evaluations)
# plot_coreset(coreset_sizes)
# plot_steps(step_sizes)
plot_metrics(train_metrics, test_metrics)