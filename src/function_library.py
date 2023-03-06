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
    """
        generate_poly_kernel_data(B_true, n, degree; inner_constant=1, outer_constant = 1, kernel_damp_normalize=true,
            kernel_damp_factor=1, noise=true, noise_half_width=0, normalize_c=true)

    Generate (X, c) from the polynomial kernel model X_{ji} ~ N(0, 1) and
    c_i(j) = ( (alpha_j * B_true[j,:] * X[:,i]  + inner_constant)^degree + outer_constant ) * epsilon_{ij} where
    alpha_j is a damping term and epsilon_{ij} is a noise term.

    # Arguments
    - `kernel_damp_normalize`: if true, then set
    alpha_j = kernel_damp_factor/norm(B_true[j,:]). This results in
    (alpha_j * B_true[j,:] * X[:,i]  + inner_constant) being normally distributed with
    mean inner_constant and standard deviation kernel_damp_factor.
    - `noise`:  if true, generate epsilon_{ij} ~  Uniform[1 - noise_half_width, 1 + noise_half_width]
    - `normalize_c`:  if true, normalize c at the end of everything
    """
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
    
# def oracle_dataset(c, oracle):
#     """
#     Apply the optimization oracle to each column of the d x n matrix c, return
#     an n vector of z_star values and a d x n matrix of optimal solutions
#     """
#     d, n = c.shape
#     z_star = np.zeros(n)
#     x_star = np.zeros((d, n))
#     for i in range(n):
#         x_star[:, i], z_star[i] = oracle(c[:, i])
#     return z_star, x_star

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
        c_hat = B_new@X[:, i]
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
############################################## Validate function #############################################
