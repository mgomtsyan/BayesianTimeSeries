import numpy as np

import torch

import pyro
import pyro.distributions as dist

from torch.distributions.normal import Normal

def log_g(x1, x2):
    d = dist.Normal(x1, x2)
    
    return d.log_prob(0.)
    
def log_likelihood(y,mu, mu_with_initial_values, gamma, gamma_with_initial_values, delta, delta_with_initial_values, z_a, z_c, sigma_e, sigma_o, sigma_u, sigma_r, sigma_v, p_a, p_c):
    
    t_not_anomaly = torch.nonzero(z_a == 0)[:,0]
    t_anomaly = torch.nonzero(z_a)[:,0]

    
    y_not_a = y[t_not_anomaly]
    y_a = y[t_anomaly]
    mu_not_a = mu[t_not_anomaly]
    mu_a = mu[t_anomaly]
    gamma_not_a = gamma[t_not_anomaly]
    gamma_a = gamma[t_anomaly]
    
    
    
    t_not_change = torch.nonzero(z_c == 0)[:,0]
    t_change = torch.nonzero(z_c)[:,0]  

    mu_not_c = mu[t_not_change]
    mu_c = mu[t_change]
    mu_prev_not_c = mu_with_initial_values[t_not_change]
    mu_prev_c = mu_with_initial_values[t_change]
    delta_prev_not_c = delta_with_initial_values[t_not_change]
    delta_prev_c = delta_with_initial_values[t_change]
                       
    delta_prev = delta_with_initial_values[:-1]           
    
    n_a = z_a.sum()
    n_not_a = torch.tensor(z_a == 0, dtype = torch.float).sum()
    
    n_c = z_c.sum()
    n_not_c = torch.tensor(z_c == 0, dtype = torch.float).sum()
    
    
    log_L =  log_g(y_not_a - mu_not_a - gamma_not_a, torch.tensor(sigma_e)).sum()

    
    log_L = log_L+log_g(y_a - mu_a - gamma_a, torch.tensor(sigma_o)).sum()

    
    log_L = log_L+log_g(mu_not_c - mu_prev_not_c - delta_prev_not_c, torch.tensor(sigma_u)).sum()

    
    log_L = log_L+log_g(mu_c - mu_prev_c - delta_prev_c, torch.tensor(sigma_r)).sum()

    
    log_L = log_L+log_g(delta - delta_prev, torch.tensor(sigma_v)).sum()

    
    log_L = log_L+ n_a*np.log(p_a ) + n_not_a *np.log(1-p_a) + n_c * np.log(p_c) + n_not_c * np.log(1-p_c)
 
    
    for t in range(len(y)):
        sum_gamma = gamma_with_initial_values[t:t+S].sum()
        
        log_L = log_L+log_g(-sum_gamma, sigma_w)    
    
    

    
    

        
    
    return log_L
