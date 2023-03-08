'''
This library includs all the functions used before the modelling and reformulation method.
Including data generation, main loop for different experiments, and some helper functions.

''' 
from model_library import train_random_forests_po, validation_set_alg, train_random_forests_po, predict_random_forests_po
from type_library import *
from model_library import *
import random
import numpy as np
import networkx as nx 
import gurobipy as gp
import matplotlib.pyplot as plt
from math import sqrt

############################################## Data Generation function #############################################

def bernoulli(p):
    if random.random() <= p:
        return 1
    else:
        return 0

def generate_poly_kernel_data(B_true, n, degree, inner_constant=1, outer_constant=1, kernel_damp_normalize=True, 
                              kernel_damp_factor=1, noise=True, noise_half_width=0, normalize_c=True, normalize_small_threshold = 0.0001):
    d, p = B_true.shape
    X_observed = np.random.normal(0, 1, (p, n))
    dot_prods = np.matmul(B_true, X_observed)

    # first generate c_observed without noise
    c_observed = np.zeros((d, n))
    for i in range(d):
        if kernel_damp_normalize:
            cur_kernel_damp_factor = kernel_damp_factor/np.linalg.norm(B_true[i,:])
        else:
            cur_kernel_damp_factor = kernel_damp_factor
        for j in range(n):
            c_observed[i, j] = (cur_kernel_damp_factor*dot_prods[i, j] + inner_constant)**degree + outer_constant
            if noise:
                epsilon = (1 - noise_half_width) + 2*noise_half_width*random.random()
                c_observed[i, j] *= epsilon
    if normalize_c:
        c_observed[:, i] = c_observed[:, i]/np.linalg.norm(c_observed[:, i])
        for j in range(d):
            if abs(c_observed[j, i]) < normalize_small_threshold:
                c_observed[j, i] = 0
    return X_observed, c_observed

def generate_poly_kernel_data_simple(B_true, n, polykernel_degree, noise_half_width):
    d, p = B_true.shape
    alpha_factor = 1/sqrt(p)
    if noise_half_width == 0:
        noise_on = False
    else:
        noise_on = True
    return generate_poly_kernel_data(B_true, n, polykernel_degree,
        kernel_damp_factor = alpha_factor,
		noise = noise_on, noise_half_width = noise_half_width,
        kernel_damp_normalize = False,
        normalize_c = False,
		inner_constant = 3)

def batch_solve(solver, y, relaxation =False):
    sol = []
    value = []
    y = np.transpose(y)
    for i in range(len(y)):
        sol.append(solver.solve(y[i], relaxation=relaxation)[0])
        value.append(solver.solve(y[i], relaxation=relaxation)[1])
    return np.transpose(np.array(value).reshape(-1, 1)), np.transpose(np.array(sol))

# Create and show the graph
def define_graph(grid_width, showflag=False):
    V = range(grid_width**2)
    E = []

    for i in V:
        if (i+1)%grid_width !=0:
            E.append((i,i+1))
        if i+grid_width<grid_width**2:
                E.append((i,i+grid_width))

    G = nx.DiGraph()
    G.add_nodes_from(V) # from point 1 to point 3
    G.add_edges_from(E)
    nx.draw_networkx(G)
    if showflag:
        plt.show()
    return G

class Shortest_path_solver:
    def __init__(self, G):
        self.G = G
    
    def make_model(self, relaxation=False):
        A = nx.incidence_matrix(self.G, oriented=True).todense() # Return a dense matrix representation of SciPy sparse matrix.
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        vtype = gp.GRB.CONTINUOUS if relaxation else gp.GRB.BINARY
        x = model.addMVar(shape=A.shape[1], name="x", vtype= vtype)
        model.addConstr(A @ x == b, name="eq")
        self.model, self.x =  model, x

    def obj_expr(self, y, x):
        return gp.quicksum(y@x)  # Sum

    def solve(self, y, relaxation=False):
        self.make_model(relaxation=relaxation)
        self.model.setObjective(y@ self.x, gp.GRB.MINIMIZE)
        self.model.optimize()
        return [self.x.x, self.model.objVal]

def convert_grid_to_list(dim1, dim2):
    G = nx.grid_2d_graph(dim1, dim2)
    sources = []
    destinations = []
    for edge in G.edges():
        sources.append(edge[0])
        destinations.append(edge[1])
    return sources, destinations

############################################## Loss function #############################################
def spo_loss(B_new, X, c, solver, z_star=[]):
    if z_star == []:
        z_star, w_star =batch_solve(solver, c)
    n = z_star.shape[1]
    spo_sum = 0
    for i in range(n):
        c_hat = np.dot(B_new, X[:, i])
        # c_hat = B_new@X[:, i]
        w_oracle = solver.solve(c_hat)[0]
        spo_loss_cur = np.dot(c[:, i], w_oracle) - z_star[:, i]
        spo_sum += spo_loss_cur
    spo_loss_avg = spo_sum / n
    return spo_loss_avg

def least_squares_loss(B_new, X, c):
    n = X.shape[1]
    residuals = np.dot(B_new, X) - c
    error = (1/n)*(1/2)*np.linalg.norm(residuals)**2
    return error

def absolute_loss(B_new, X, c):
    n = X.shape[1]
    residuals = np.dot(B_new, X) - c
    error = (1/n)*np.linalg.norm(residuals, 1)
    return error

def spo_plus_loss(B_new, X, c, solver, z_star=[], w_star=[]):
    if z_star == [] or w_star == []:
        z_star, w_star =batch_solve(solver, c)
    spo_plus_sum = 0
    z_star = np.transpose(np.array(z_star))
    n = X.shape[0]
    for i in range(n):
        c_hat = B_new @ X[:, i]
        spoplus_cost_vec = 2 * c_hat - c[:, i]
        z_oracle= solver.solve(spoplus_cost_vec)[1]

        spo_plus_cost = -z_oracle + 2 * np.dot(c_hat, w_star[:, i]) - z_star[i]
        spo_plus_sum += spo_plus_cost

    spo_plus_avg = spo_plus_sum / n
    return spo_plus_avg

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