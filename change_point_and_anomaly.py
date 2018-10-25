import numpy as np

import torch

import pyro
import pyro.distributions as dist

from torch.distributions.normal import Normal

def update_p_a_t(p_a, sigma_0, sigma_e, y_t, mu_t, gamma_t):
    distr_1 = dist.Normal(mu_t + gamma_t,sigma_0)
    distr_0 = dist.Normal(mu_t + gamma_t,sigma_e)
    p_a_t_1 = p_a * torch.exp(distr_1.log_prob(y_t))
    p_a_t_0 = (1 - p_a) * torch.exp(distr_0.log_prob(y_t))
    
    p_a_t = p_a_t_1 / (p_a_t_1 + p_a_t_0)
    
    return p_a_t
    
def update_p_c_t(p_c, sigma_r, sigma_u, mu_t, mu_t_prev, delta_t_prev):
    distr_1 = dist.Normal(mu_t_prev + delta_t_prev,sigma_r)
    distr_0 = dist.Normal(mu_t_prev + delta_t_prev,sigma_u)
    p_c_t_1 = p_c * torch.exp(distr_1.log_prob(mu_t))
    p_c_t_0 = (1 - p_c) * torch.exp(distr_0.log_prob(mu_t))
    
    p_c_t = p_c_t_1 / (p_c_t_1 + p_c_t_0)
    
    return p_c_t
    
def generate_z_a_t(p_a, sigma_0, sigma_e, y_t, mu_t, gamma_t):
    p_a_t = update_p_a_t(p_a, sigma_0, sigma_e, y_t, mu_t, gamma_t)
    distr_1 = distr.Bernoulli(p_a_t)
    
    return distr_1.sample()
    
def generate_z_c_t(p_c, sigma_r, sigma_u, mu_t, mu_t_prev, delta_t_prev):
    p_c_t = update_p_a_t(p_c, sigma_r, sigma_u, mu_t, mu_t_prev, delta_t_prev)
    distr_1 = distr.Bernoulli(p_c_t)
    
    return distr_1.sample()
    
def segment_control(z_c, mu, sigma_r, l):

    t_change_points = torch.nonzero(z_c)[:,0]
    
    distance_between_change_points = t_change_points[1:] - t_change_points[:-1]
  
    bad_points = distance_between_change_points < l
    
    if bad_points.sum() == 0:
        return z_c
    
    idx_too_close_pairs_of_points = torch.nonzero(distance_between_change_points < l)[:,0]
     
    while bad_points.sum() > 0:
        
        idx_too_close_pairs_of_points = torch.nonzero(distance_between_change_points < l)[:,0]
        
        idx_point_1 = idx_too_close_pairs_of_points[0]
        idx_point_2 = idx_point_1 + 1
        
        t_point_1 = t_change_points[idx_point_1]
        t_point_2 = t_change_points[idx_point_2]
        

        if torch.abs(mu[t_point_1-1] - mu[t_point_2 + 1]) <= sigma_r / 2:
            
            z_c[t_point_1] = 0
            z_c[t_point_2] = 0
            
            print('exclude both')
            
        else:
            
            i_exclude = np.random.randint(1,3)
            
            print('exclude', i_exclude)
            
            if i_exclude == 1:
                z_c[t_point_1] = 0
            else:
                z_c[t_point_2] = 0
                
        t_change_points = torch.nonzero(z_c)[:,0]

        distance_between_change_points = t_change_points[1:] - t_change_points[:-1]
        
        bad_points = distance_between_change_points < l

    return z_c
