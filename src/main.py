from function_library import *

# Big size test
# Fixed parameter settings (these never change)
p_features = 5
grid_dim = 5
num_trials = 3
n_test = 10000
num_lambda = 5
lambda_max = 0
lambda_min_ratio = 1
holdout_percent = 0.25
regularization = 'lasso'
different_validation_losses = False
include_rf = True

# Fixed parameter sets (these are also the same for all experiments)
n_train_vec = [5000]
polykernel_degree_vec = [8]
polykernel_noise_half_width_vec = [0, 0.5]
# Set this to get reproducible results
rng_seed = 5352
expt_results_large = shortest_path_multiple_replications(rng_seed, num_trials, grid_dim,
    n_train_vec, n_test,
    p_features, polykernel_degree_vec, polykernel_noise_half_width_vec,
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio,
    holdout_percent = holdout_percent, regularization = regularization,
    different_validation_losses = different_validation_losses)
expt_results_large['SPOplus_spoloss_test'] = expt_results_large['SPOplus_spoloss_test'].apply(lambda x: pd.Series(x).astype(float))
expt_results_large['LS_spoloss_test'] = expt_results_large['LS_spoloss_test'].apply(lambda x: pd.Series(x).astype(float))
expt_results_large['RF_spoloss_test'] = expt_results_large['RF_spoloss_test'].apply(lambda x: pd.Series(x).astype(float))
expt_results_large['Absolute_spoloss_test'] = expt_results_large['Absolute_spoloss_test'].apply(lambda x: pd.Series(x).astype(float))
csv_string = "shortest_path_50000.csv"
expt_results_large.to_csv(csv_string, index=False)