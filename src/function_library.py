'''
This library includs all the functions used before the modelling and reformulation method.
Including data generation, main loop for different experiments, and some helper functions.

''' 
from helper_library import *
import numpy as np
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor

################################################### Main loop ###################################################
def shortest_path_replication(grid_dim, n_train, n_holdout, n_test,p_features, polykernel_degree, polykernel_noise_half_width, 
                              num_lambda = 10, lambda_max = None, lambda_min_ratio = 0.0001, regularization = 'ridge', 
                              different_validation_losses = False, include_rf = True):

    # Set up data
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

    # c_bar_test_preds = np.zeros((d_feasibleregion, n_test))
    # for i in range(n_test):
    #     c_bar_test_preds[:, i] = c_bar_train
    # final_results.Baseline_spoloss_test = spo_loss(np.matrix(np.eye(d_feasibleregion)), c_bar_test_preds, c_test, solver)

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

############################################## Reformulation function #############################################
def sp_reformulation_path(X, c, solver, sp_graph, path_alg_parms):

    path_alg_parms = PathParms().__dict__
    sp_graph_parms = sp_graph
    sources = sp_graph_parms['sources']
    destinations = sp_graph_parms['destinations']
    grid_dim = sp_graph_parms['grid_dim']
    lambda_max = path_alg_parms['lambda_max']
    lambda_min_ratio = path_alg_parms['lambda_min_ratio']
    num_lambda = path_alg_parms['num_lambda']
    regularization = path_alg_parms['regularization']

    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")
    # Process graph
    nodes = set(sources) | set(destinations)
    n_nodes = len(nodes)
    n_edges = len(sources)
    if n_edges != d:
        raise ValueError("Dimensions of the input are mismatched.")
    # Hard code sparse incidence matrix!
    A, b = CreateShortestPathConstraints(grid_dim)

    X = X.T
    c = c.T
    solver = ShortestPathSolver(A, b)
    W = np.apply_along_axis(solver.solve, 1, c)#W has shape [num_samples, num_edges]
    #define linear program variables
    B = cp.Variable((d, p)) #B has shape [num_edges, num_features]
    P = cp.Variable((n, n_nodes), nonneg = True) #P has shape [num_samples, num_nodes]
    objective = (cp.sum(-P@b) + 2*cp.sum(cp.multiply(X@B.T, W)) - cp.sum(cp.multiply(W, c)))
    # Add regularization part
    if lambda_max is None:
        lambda_max = (d/n)*(np.linalg.norm(X)**2)

    # Construct lambda sequence
    if num_lambda == 1 and lambda_max == 0:
        lambdas = [0.0]
    else:
        lambda_min = lambda_max*lambda_min_ratio
        log_lambdas = np.linspace(np.log(lambda_min), np.log(lambda_max), num=num_lambda)
        lambdas = np.exp(log_lambdas)
    
    # Add theta variables for Lasso
    if regularization == 'lasso':
        theta = cp.Variable((d, p))
    B_soln_list = []

    for lambda_ in lambdas:
        if regularization == 'lasso':
            objective += n*lambda_*cp.sum(theta)
            prob = cp.Problem(cp.Minimize(objective), [(P@A) <= ((2*(X@B.T)) - c), theta >= B, theta >= -B])
        else:
            objective += n*(lambda_/2)*cp.atoms.norm(B, 'fro')
            prob = cp.Problem(cp.Minimize(objective), 
                                        [(P@A) <= ((2*(X@B.T)) - c)])
        prob.solve()
        B_matrix = B.value
        if B_matrix is None:
            B_matrix = np.zeros((d, p))
        B_soln_list.append(B_matrix)

    return B_soln_list, lambdas

def leastSquares_path_jump(X, c, solver, sp_graph, path_alg_parms):
    path_alg_parms = PathParms().__dict__
    sp_graph_parms = sp_graph
    sources = sp_graph_parms['sources']
    destinations = sp_graph_parms['destinations']
    grid_dim = sp_graph_parms['grid_dim']
    lambda_max = path_alg_parms['lambda_max']
    lambda_min_ratio = path_alg_parms['lambda_min_ratio']
    num_lambda = path_alg_parms['num_lambda']
    po_loss_function = path_alg_parms['po_loss_function']
    regularization = path_alg_parms['regularization']

    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")
    # Process graph
    nodes = set(sources) | set(destinations)
    n_nodes = len(nodes)
    n_edges = len(sources)
    if n_edges != d:
        raise ValueError("Dimensions of the input are mismatched.")
    # Hard code sparse incidence matrix!
    A, b = CreateShortestPathConstraints(grid_dim)

    X = X.T
    c = c.T
    solver = ShortestPathSolver(A, b)
    W = np.apply_along_axis(solver.solve, 1, c)#W has shape [num_samples, num_edges]
    #define linear program variables
    B = cp.Variable((d, p)) #B has shape [num_edges, num_features]

    if po_loss_function == 'leastSquares':
        obj_expr_noreg = cp.sum_squares(c-X@B.T)
        solver='SCS'
        constraint = []
    elif po_loss_function == 'absolute':
        w = cp.Variable((d, n), nonneg = True)
        constraint =[c-X@B.T <= w.T, -(c-X@B.T) <= w.T]
        # @constraint(mod, c - B_var*X .<= w_var)
        # @constraint(mod, -(c - B_var*X) .<= w_var)
        solver = 'ECOS'
        obj_expr_noreg = 2*cp.sum(cp.multiply(w, np.ones((d, n))))

    # Add regularization part
    if lambda_max is None:
        lambda_max = (d/n)*(np.linalg.norm(X)**2)

    # Construct lambda sequence
    if num_lambda == 1 and lambda_max == 0:
        lambdas = [0.0]
    else:
        lambda_min = lambda_max*lambda_min_ratio
        log_lambdas = np.linspace(np.log(lambda_min), np.log(lambda_max), num=num_lambda)
        lambdas = np.exp(log_lambdas)
    
    # Add theta variables for Lasso
    if regularization == 'lasso':
        theta = cp.Variable((d, p))

    B_soln_list = []

    for lambda_ in lambdas:
        if regularization == 'lasso':
            obj_expr_full = obj_expr_noreg + 2*n*lambda_*cp.sum(theta)
            prob = cp.Problem(cp.Minimize(obj_expr_full), constraint + [theta >= B, theta >= -B])
        else:
            obj_expr_full = obj_expr_noreg + n*(lambda_/2)*cp.atoms.norm(B, 'fro')
            prob = cp.Problem(cp.Minimize(obj_expr_full), constraint)
        prob.solve(solver=solver)
        B_matrix = B.value
        if B_matrix is None:
            B_matrix = np.zeros((d, p))
        B_soln_list.append(B_matrix)
    return B_soln_list, lambdas

def validation_set_alg(X_train, c_train, X_validation, c_validation, solver, sp_graph, val_alg_parms, path_alg_parms):
    val_alg_parms = val_alg_parms.__dict__
    path_alg_parms = path_alg_parms.__dict__
    sp_graph = sp_graph.__dict__
    algorithm_type = val_alg_parms['algorithm_type']
    validation_loss = val_alg_parms['validation_loss']
    if algorithm_type == 'sp_spo_plus_reform':
        B_soln_list, lambdas = sp_reformulation_path(X_train, c_train, solver, sp_graph, path_alg_parms)
    elif algorithm_type == 'ls_jump':
        B_soln_list, lambdas = leastSquares_path_jump(X_train, c_train, solver, sp_graph, path_alg_parms)
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros(num_lambda)
    for i in range(num_lambda):
        if validation_loss == "spo_loss":
            validation_loss_list[i] = spo_loss(B_soln_list[i], X_validation, c_validation, solver)
        elif validation_loss == "spo_plus_loss":
            validation_loss_list[i] = spo_plus_loss(B_soln_list[i], X_validation, c_validation, solver)
        elif validation_loss == "least_squares_loss":
            validation_loss_list[i] = least_squares_loss(B_soln_list[i], X_validation, c_validation)
        elif validation_loss == "absolute_loss":
            validation_loss_list[i] = absolute_loss(B_soln_list[i], X_validation, c_validation)
        else:
            raise ValueError("Enter a valid validation set loss function.")
    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    best_B_matrix = B_soln_list[best_ind]
    return best_B_matrix, best_lambda

################################################## Decision Tree functions ########################################

def train_random_forests_po(X: np.ndarray, c: np.ndarray, rf_alg_parms=rf_graph()):
    num_trees = rf_alg_parms.__dict__['num_trees']
    num_features_per_split = rf_alg_parms.__dict__['num_features_per_split']

    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")
    X_t = X.T
    # If num_features_per_split is not specified, use default for regression
    if num_features_per_split < 1:
        num_features_per_split = int(np.ceil(p/3))

    rf_model_list = []

    # train one model for each component
    for j in range(d):
        c_vec = c[j, :]
        rf_model = RandomForestRegressor(n_estimators=num_trees, max_features=num_features_per_split)
        rf_model.fit(X_t, c_vec)
        rf_model_list.append(rf_model)

    return rf_model_list

def predict_random_forests_po(rf_model_list, X_new):
    p, n = X_new.shape
    d = len(rf_model_list)
    X_new_t = X_new.T
    preds = np.zeros((d, n))
    for j in range(d):
        preds[j, :] = rf_model_list[j].predict(X_new_t)
    return preds