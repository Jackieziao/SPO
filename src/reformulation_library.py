from dataclasses import dataclass
from typing import Union
import attr
from typing import List
from function_library import *
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor


@attr.s(auto_attribs=True)
class replication_results:
    SPOplus_spoloss_test: Union[float, None] = None
    LS_spoloss_test: Union[float, None] = None
    RF_spoloss_test: Union[float, None] = None
    Absolute_spoloss_test: Union[float, None] = None
    Baseline_spoloss_test: Union[float, None] = None
    zstar_avg_test: Union[float, None] = None

@dataclass
class shortest_path_graph:
    sources: List[int]
    destinations: List[int]
    start_node: int 
    end_node: int
    acyclic: bool = True
    grid_dim : int = 5

@dataclass()
class ValParms:
    algorithm_type: str = 'sp_spo_plus_reform'
    validation_set_percent: float = 0.2
    validation_loss: str = 'spo_loss'
    plot_results: bool = False
    resolve_sgd: bool = False
    resolve_sgd_accuracy: float = 0.00001
    resolve_iteration_limit: int = 50000

@dataclass()
class PathParms:
    lambda_max: None = None
    lambda_min_ratio: float = 0.0001
    num_lambda: int = 10
    solver: str = "Gurobi"
    regularization: str = "ridge"
    regularize_first_column_B: bool = False
    upper_bound_B_present: bool = False
    upper_bound_B: float = 10.0**6
    po_loss_function: str = "leastSquares"
    verbose: bool = False
    algorithm_type: str = "fake_algorithm"

@dataclass
class rf_graph:
    num_trees : int = 100
    num_features_per_split : int = -1

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

def sp_reformulation_path(X, c, solver, sp_graph, path_alg_parms):

    path_alg_parms = PathParms().__dict__
    sp_graph_parms = sp_graph
    sources = sp_graph_parms['sources']
    destinations = sp_graph_parms['destinations']
    grid_dim = sp_graph_parms['grid_dim']
    lambda_max = path_alg_parms['lambda_max']
    lambda_min_ratio = path_alg_parms['lambda_min_ratio']
    num_lambda = path_alg_parms['num_lambda']

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
    B_soln_list = []

    for lambda_ in lambdas:
        objective += n*(lambda_/2)*cp.atoms.norm(B, 'fro')
        prob = cp.Problem(cp.Minimize(objective), 
                                    [(P@A) <= ((2*(X@B.T)) - c)])
        prob.solve()
        B_soln_list.append(B.value)

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
    B_soln_list = []

    for lambda_ in lambdas:
        obj_expr_full = obj_expr_noreg + n*(lambda_/2)*cp.atoms.norm(B, 'fro')
        prob = cp.Problem(cp.Minimize(obj_expr_full), constraint)
        prob.solve(solver=solver)
        B_soln_list.append(B.value)
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