'''
This library includs some parameter data structure and dataframe of the final result.
Also some helper function for main library.

''' 
from dataclasses import dataclass
from typing import Union
import attr
from typing import List
import pandas as pd
import random
import numpy as np
import networkx as nx 
import gurobipy as gp
import matplotlib.pyplot as plt
from math import sqrt
import cvxpy as cp

######################################## Data structure ########################################

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
    regularization: str = "lasso"
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

######################################## Dataframe structure ########################################

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
        z_star, w_star = batch_solve(solver, c)
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