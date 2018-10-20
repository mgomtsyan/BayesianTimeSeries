import numpy as np

import torch

import pyro
import pyro.distributions as dist

from torch.distributions.normal import Normal

def update_sigma_e(y, mu, gamma, z_a, sigma_e):
    
    if (z_a == 0).sum() == 0:
        return sigma_e
    
    t_not_anomaly = torch.nonzero(z_a == 0)[:,0]
    n = t_not_anomaly.shape[0]
    
    y_not_a = y[t_not_anomaly]
    mu_not_a = mu[t_not_anomaly]
    gamma_not_a = gamma[t_not_anomaly]
    
    sigma_e = ( ((y_not_a - mu_not_a - gamma_not_a) **2).sum() / n)** 0.5
    
    return sigma_e
    
def update_sigma_o(y, mu, gamma, z_a, sigma_o):
    
    if z_a.sum() == 0:
        return z_a
    
    t_anomaly = torch.nonzero(z_a)[:,0]
    n = t_anomaly.shape[0]
    
    y_a = y[t_anomaly]
    mu_a = mu[t_anomaly]
    gamma_a = gamma[t_anomaly]
    
    sigma_o = ( ((y_a - mu_a - gamma_a) **2).sum() / n)** 0.5
    
    return sigma_o
    
def update_sigma_u(mu,mu_with_initial_values, delta, delta_with_initial_values, z_c, sigma_u):
    
    if (z_c == 0).sum() == 0:
        return sigma_u
    
    t_not_change = torch.nonzero(z_c == 0)[:,0]
    
    n = t_not_change.shape[0]
    
    mu_not_c = mu[t_not_change]
    mu_prev_not_c = mu_with_initial_values[t_not_change]
    delta_prev_not_c = delta_with_initial_values[t_not_change]
    
    sigma_u = ( ((mu_not_c - mu_prev_not_c - delta_prev_not_c) **2).sum() / n)** 0.5
    
    return sigma_u
    
def update_sigma_r(mu,mu_with_initial_values, delta, delta_with_initial_values, z_c, sigma_r):
    
    if z_c.sum() == 0:
        return sigma_r
    
    t_change = torch.nonzero(z_c)[:,0]
    t_prev_change = t_change - 1
    
    n = t_change.shape[0]
    
    mu_c = mu[t_change]
    mu_prev_c = mu_with_initial_values[t_change]
    delta_prev_c = delta_with_initial_values[t_change]
    
    sigma_r = ( ((mu_c - mu_prev_c - delta_prev_c) **2).sum() / n)** 0.5
    
    return sigma_r
    
def update_sigma_delta(delta, delta_with_initial_values):
    n = delta.shape[0] 
    sigma_delta = (((delta - delta_with_initial_values[:-1]) ** 2).sum() / n) ** 0.5
    
    return sigma_delta
    
def update_sigma_gamma(gamma, gamma_with_initial_values, S):
    n = gamma.shape[0]

    
    sum_sum_g_2 = 0
    for t in range(len(gamma)):
        sum_g_2 = (gamma_with_initial_values[t:t+S-1].sum())**2
        
        sum_sum_g_2 += sum_g_2
    
    sigma_gamma = (sum_sum_g_2 / n) ** 0.5
    
    return(sigma_gamma)
