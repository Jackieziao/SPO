import pickle
import random
import numpy as np
import networkx as nx 
import gurobipy as gp
from torch import nn
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

############################################## Loss function #############################################
def reformulation():
    return None


############################################## Loss function #############################################
def spo_plus_loss():
    return None

def least_squares_loss(y_hat, y):
    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(y_hat, y)
    return loss

def absolute_loss():
    return None

def random_forests():
    return None