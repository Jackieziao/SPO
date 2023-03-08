from function_library import *
from reformulation_library import *
from main_library import *

# Small size test
# Fixed parameter settings (these never change)
# Small size test
# Fixed parameter settings (these never change)
p_features = 5
grid_dim = 5
num_trials = 3
n_test = 10000

num_lambda = 10
lambda_max = 100
lambda_min_ratio = 10 ** (-8)
holdout_percent = 0.25
regularization = 'lasso'
different_validation_losses = False


# Fixed parameter sets (these are also the same for all experiments)
n_train_vec = [100, 1000]
polykernel_degree_vec = [1, 2, 4, 6, 8]
polykernel_noise_half_width_vec = [0, 0.5]

# Set this to get reproducible results
rng_seed = 5352
expt_results = shortest_path_multiple_replications(rng_seed, num_trials, grid_dim,
    n_train_vec, n_test,
    p_features, polykernel_degree_vec, polykernel_noise_half_width_vec,
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
    holdout_percent = holdout_percent, regularization = regularization,
    different_validation_losses = different_validation_losses)

# Big size test
# Fixed parameter settings (these never change)
p_features = 5
grid_dim = 5
num_trials = 3
n_test = 10000

num_lambda = 1
lambda_max = 0
lambda_min_ratio = 1
holdout_percent = 0.25
regularization = 'lasso'
different_validation_losses = False
include_rf = True


# Fixed parameter sets (these are also the same for all experiments)
n_train_vec = [5000]
polykernel_degree_vec = [1, 2, 4, 6, 8]
polykernel_noise_half_width_vec = [0, 0.5]
# Set this to get reproducible results
rng_seed = 5352
expt_results_large = shortest_path_multiple_replications(rng_seed, num_trials, grid_dim,
    n_train_vec, n_test,
    p_features, polykernel_degree_vec, polykernel_noise_half_width_vec,
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
    holdout_percent = holdout_percent, regularization = regularization,
    different_validation_losses = different_validation_losses)