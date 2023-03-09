'''
This library including all the functions used for the modelling and reformulation method.

''' 
from type_library import *
from model_library import batch_solve
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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
        z_star, w_star = batch_solve(solver, c)
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

############################################## Helper function #############################################
class ShortestPathSolver:
    def __init__(self,A,b):
        '''
        Defines binary optimization problem to solve the shortest path problem with constraint matrix A and RHS b as numpy arrays
        
        Parameters:
            np.array A: constraint matrix A
            np.array b: RHS of constraints
        '''
        if A.shape[0] != b.size:
            print('invalid input')
            return
        numedges = A.shape[1]
        self.c = cp.Parameter(numedges)
        self.w = cp.Variable(numedges, nonneg=True)
        self.prob = cp.Problem(cp.Minimize(self.c@self.w), 
                               [A @ self.w == b, self.w <= 1]) #add a trivial inequality constraint because necessary for GLPK_MI solver
        
    def solve(self,c):
        '''
        Solves the predefined optmiization problem with cost vector c and returns the decision variable array
        
        Parameters:
            np.array c: cost vector 
            np.array b: RHS of constraints
        
        Returns:
            np.array of the solution to the shortest path problem
        '''
        self.c.project_and_assign(c)
        self.prob.solve()
        return self.w.value

def CreateShortestPathConstraints(gridsize):
    '''
    Generate constraints for the nxn grid shortest path problem. 
    Each node in the grid has a constraint where the LHS is the inflows - outflows and the RHS is the desired flow.
    The desired flow is 0 for all nodes except for the start node where it's -1 and end node where it's 1
    
    Parameters:
        int gridsize: Size of each dimension in grid
        
    Returns:
        np.array A: Flow matrix of shape [num_nodes, num_edges]. Aij is -1 if the edge j is an outflow of node i and 1 if edge edge j is an inflow of node i
        np.array b: RHS of constraints [num_nodes]
    '''
    # define node and edge sizes
    num_nodes = gridsize**2
    num_directional_edges = (num_nodes - gridsize) # num vertical edges and num horizontal edges
    num_edges = num_directional_edges*2 # sum vertical and horizontal edges together
    
    # initialize empty A and B arrays
    A = np.zeros((num_nodes, num_edges), np.int8)
    b = np.zeros(num_nodes, np.int8)
    
    # fill in flow matrix
    # nodes are ordered by rows. ex. in a 3x3 grid the first rows nodes are indices 1,2,3 and second row is 4,5,6
    # horizontal edges are enumerated first and then vertical edges
    horizontaledgepointer = 0
    verticaledgepointer = 0
    for i in range(num_directional_edges):
        # update flow matrix for horizontal edges
        outnode = horizontaledgepointer
        innode = horizontaledgepointer + 1
        
        A[outnode, i] = -1
        A[innode, i] = 1
        horizontaledgepointer += 1
        if (horizontaledgepointer + 1)% gridsize == 0:# node is at right edge of the grid so  go to next row
            horizontaledgepointer += 1
        
        # update flow matrix for vertical edges
        outnode = verticaledgepointer
        innode = verticaledgepointer + gridsize
        A[outnode, num_directional_edges + i] = -1
        A[innode, num_directional_edges + i] = 1
        verticaledgepointer += gridsize
        if verticaledgepointer + gridsize >= num_nodes:# node is at bottom edge of the grid so go to next column
            verticaledgepointer = (verticaledgepointer % gridsize) + 1
        
    # update RHS for start and end nodes
    b[0] = -1
    b[-1] = 1 
    return A, b

def ridge(X, c, reg_param):
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("dimensions are mismatched")
    Xt = X.T
    ct = c.T
    Bt = np.linalg.inv(X @ Xt + n * reg_param * np.eye(p)) @ (X @ ct)
    return Bt.T

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