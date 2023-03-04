import numpy as np
from numpy.linalg import inv

def cost_least_squares(X, c, intercept=False):
    p, n = X.shape
    d, n2 = c.shape

    if n != n2:
        raise ValueError("dimensions are mismatched")

    if intercept:
        X = np.vstack((X, np.ones((1, n))))

    Xt = X.T
    ct = c.T

    Bt = inv(Xt @ X) @ Xt @ ct

    return Bt.T


def ridge(X, c, reg_param):
    p, n = X.shape
    d, n2 = c.shape

    if n != n2:
        raise ValueError("dimensions are mismatched")

    Xt = X.T
    ct = c.T

    Bt = inv(X @ Xt + n * reg_param * np.eye(p)) @ X @ ct

    return Bt.T


def oracle_dataset(c, oracle):
    d, n = c.shape
    z_star_data = np.zeros(n)
    w_star_data = np.zeros((d, n))
    for i in range(n):
        z_i, w_i = oracle(c[:, i])
        z_star_data[i] = z_i
        w_star_data[:, i] = w_i
    return z_star_data, w_star_data


def spo_loss(B_new, X, c, oracle, z_star=[]):
    if len(z_star) == 0:
        z_star, w_star = oracle_dataset(c, oracle)

    n = len(z_star)

    spo_sum = 0
    for i in range(n):
        c_hat = B_new @ X[:, i]
        z_oracle, w_oracle = oracle(c_hat)
        spo_loss_cur = np.dot(c[:, i], w_oracle) - z_star[i]
        spo_sum = spo_sum + spo_loss_cur

    spo_loss_avg = spo_sum / n
    return spo_loss_avg


def compare_models_percent(B_modA, B_modB, X, c, oracle, eps=0.000001):
    d, n = c.shape

    count = 0
    for i in range(n):
        c_hatA = B_modA @ X[:, i]
        c_hatB = B_modB @ X[:, i]

        z_oracleA, w_oracleA = oracle(c_hatA)
        z_oracleB, w_oracleB = oracle(c_hatB)

        diff = np.dot(c[:, i], w_oracleA) - np.dot(c[:, i], w_oracleB)
        if diff <= eps:
            count = count + 1

    return count / n


def spo_plus_loss(B_new, X, c, oracle, z_star=[], w_star=[]):
    if len(z_star) == 0:
        z_star, w_star = oracle_dataset(c, oracle)

    n = len(z_star)

    spo_sum = 0
    for i in range(n):
        c_hat = B_new @ X[:, i]
        z_oracle, w_oracle = oracle(c_hat)
        spo_plus_loss_cur = np.dot(c[:, i], w_oracle) - z_star[i] + np.linalg.norm(w_oracle - w_star[:, i])**2
        spo_sum = spo_sum + spo_plus_loss_cur

    spo_plus_loss_avg = spo_sum / n
    return spo_plus_loss_avg

def least_squares_loss(B_new, X, c):
    p, n = X.shape
    residuals = B_new @ X - c
    error = (1 / n) * (1 / 2) * (np.linalg.norm(residuals)**2)
    return error

def absolute_loss(B_new, X, c):
    p, n = X.shape
    residuals = B_new @ X - c
    error = (1 / n) * np.linalg.norm(residuals, 1)
    return error