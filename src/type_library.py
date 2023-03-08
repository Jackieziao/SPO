'''
This library includs some parameter data structure and dataframe of the final result..

''' 
from type_library import *
from dataclasses import dataclass
from typing import Union
import attr
from typing import List
import pandas as pd

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