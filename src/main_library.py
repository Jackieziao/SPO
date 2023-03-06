from function_library import *
from reformulation_library import *
import pandas as pd

def make_blank_complete_df():
    results_df = pd.DataFrame({
        'grid_dim': [],
        'n_train': [],
        'n_holdout': [],
        'n_test': [],
        'p_features': [],
        'polykernel_degree': [],
        'polykernel_noise_half_width': [],
        'SPOplus_spoloss_test': [],
        'LS_spoloss_test': [],
        'RF_spoloss_test': [],
        'Absolute_spoloss_test': [],
        'zstar_avg_test': []
    })

    return results_df

def build_complete_row(grid_dim, n_train, n_holdout, n_test, p_features, polykernel_degree, polykernel_noise_half_width, results_struct):
    df = make_blank_complete_df()
    df.loc[0] = [
        grid_dim,
        n_train,
        n_holdout,
        n_test,
        p_features,
        polykernel_degree,
        polykernel_noise_half_width,
        results_struct.SPOplus_spoloss_test,
        results_struct.LS_spoloss_test,
        results_struct.RF_spoloss_test,
        results_struct.Absolute_spoloss_test,
        results_struct.zstar_avg_test
    ]

    return df
  
def shortest_path_replication(grid_dim, n_train, n_holdout, n_test,p_features, polykernel_degree, polykernel_noise_half_width, 
                              num_lambda = 10, lambda_max = None, lambda_min_ratio = 0.0001, regularization = 'ridge', 
                              different_validation_losses = False, include_rf = True):

    different_validation_losses = False
    n_holdout = 1000
    n_test = 1000
    grid_dim = 5
    p_features = 5
    n_train = 1000
    polykernel_degree = 1
    polykernel_noise_half_width = 0
    d_feasibleregion = 2 * p_features * (p_features - 1)
    sources, destinations = convert_grid_to_list(grid_dim, grid_dim)
    sp_graph = shortest_path_graph(sources = sources, destinations = destinations,
            start_node = 1, end_node = grid_dim^2, acyclic = True)
    B_true = np.array([[bernoulli(0.5) for k in range(p_features)] for e in range(d_feasibleregion)])
    X_train, c_train = generate_poly_kernel_data_simple(B_true, n_train, polykernel_degree, polykernel_noise_half_width)
    X_validation, c_validation = generate_poly_kernel_data_simple(B_true, n_holdout, polykernel_degree, polykernel_noise_half_width)
    X_test, c_test = generate_poly_kernel_data_simple(B_true, n_test, polykernel_degree, polykernel_noise_half_width)

    # Solve the shortest path problem
    G = define_graph(grid_dim, showflag=False)
    solver = Shortest_path_solver(G)
    z_test, w_test = batch_solve(solver, c_test)

    # Set validation losses
    if different_validation_losses:
        spo_plus_val_loss = 'spo_loss'
        ls_val_loss = 'least_squares_loss'
        absolute_val_loss = 'absolute_loss'
    else:
        spo_plus_val_loss = 'spo_loss'
        ls_val_loss = 'spo_loss'
        absolute_val_loss = 'spo_loss'
    ### Algorithms ###

    # SPO+
    best_B_SPOplus, best_lambda_SPOplus = validation_set_alg(X_train, c_train, X_validation, c_validation, solver, sp_graph = sp_graph,
        val_alg_parms = ValParms(algorithm_type = 'sp_spo_plus_reform', validation_loss = spo_plus_val_loss),
        path_alg_parms = PathParms(num_lambda = num_lambda, lambda_max = lambda_max, 
                                                regularization = regularization,
                                                lambda_min_ratio = lambda_min_ratio, algorithm_type = 'SPO_plus'))
    best_B_leastSquares, best_lambda_leastSquares = validation_set_alg(X_train, c_train, X_validation, c_validation, solver, sp_graph = sp_graph,
        val_alg_parms = ValParms(algorithm_type = 'ls_jump', validation_loss = ls_val_loss),
        path_alg_parms = PathParms(num_lambda = num_lambda, lambda_max = lambda_max, 
                                                regularization = regularization,
                                                lambda_min_ratio = lambda_min_ratio, algorithm_type = 'leastSquares'))
    best_B_Absolute, best_lambda_Absolute = validation_set_alg(X_train, c_train, X_validation, c_validation, solver, sp_graph = sp_graph,
        val_alg_parms = ValParms(algorithm_type = 'ls_jump', validation_loss = absolute_val_loss),
        path_alg_parms = PathParms(num_lambda = num_lambda, lambda_max = lambda_max, 
                                                regularization = regularization,
                                                po_loss_function = 'absolute',
                                            lambda_min_ratio = lambda_min_ratio, algorithm_type = 'Absolute'))
    # RF
    if include_rf:
        rf_mods = train_random_forests_po(X_train, c_train, rf_alg_parms=rf_graph())
    
    # Baseline
    c_bar_train = np.mean(c_train, axis=1, keepdims=True)

    ### Populate final results ###
    final_results = replication_results()

    final_results.SPOplus_spoloss_test = spo_loss(best_B_SPOplus, X_test, c_test, solver)
    final_results.LS_spoloss_test = spo_loss(best_B_leastSquares, X_test, c_test, solver)
    final_results.Absolute_spoloss_test = spo_loss(best_B_Absolute, X_test, c_test, solver)
    c_bar_test_preds = np.zeros((d_feasibleregion, n_test))

    if include_rf:
        rf_preds_test = predict_random_forests_po(rf_mods, X_test)
        final_results.RF_spoloss_test = spo_loss(np.matrix(np.eye(d_feasibleregion)), rf_preds_test, c_test, solver)
    else:
        final_results.RF_spoloss_test = None

    c_bar_test_preds = np.zeros((d_feasibleregion, n_test))
    for i in range(n_test):
        c_bar_test_preds[:, i] = c_bar_train
    final_results.Baseline_spoloss_test = spo_loss(np.matrix(np.eye(d_feasibleregion)), c_bar_test_preds, c_test, solver)

    final_results.zstar_avg_test = np.mean(z_test)

    return final_results

def shortest_path_multiple_replications(rng_seed, num_trials, grid_dim, n_train_vec, n_test, p_features, polykernel_degree_vec, polykernel_noise_half_width_vec, num_lambda=10, lambda_max=None, lambda_min_ratio=0.0001, holdout_percent=0.25, regularization='ridge', different_validation_losses=False, include_rf=True):
    np.random.seed(rng_seed)
    big_results_df = make_blank_complete_df()

    for n_train in n_train_vec:
        for polykernel_degree in polykernel_degree_vec:
            for polykernel_noise_half_width in polykernel_noise_half_width_vec:
                print(f"Moving on to n_train = {n_train}, polykernel_degree = {polykernel_degree}, polykernel_noise_half_width = {polykernel_noise_half_width}")
                for trial in range(num_trials):
                    print(f"Current trial is {trial}")
                    n_holdout = round(holdout_percent*n_train)

                    current_results = shortest_path_replication(grid_dim,
                        n_train, n_holdout, n_test,
                        p_features, polykernel_degree, polykernel_noise_half_width,
                        num_lambda=num_lambda, lambda_max=lambda_max, lambda_min_ratio=lambda_min_ratio,
                        regularization=regularization,
                        different_validation_losses=different_validation_losses,
                        include_rf=include_rf)

                    current_df_row = build_complete_row(grid_dim,
                        n_train, n_holdout, n_test,
                        p_features, polykernel_degree, polykernel_noise_half_width, current_results)

                    big_results_df = pd.concat([big_results_df, current_df_row])

    return big_results_df
