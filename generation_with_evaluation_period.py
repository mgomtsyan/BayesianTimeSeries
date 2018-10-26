import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
import pyro.distributions as dist

from torch.distributions.normal import Normal

def generative_procedure_with_evaluation_period(mu_0, delta_0, gamma_0, sigma_e, sigma_o, sigma_u, sigma_r, sigma_v, sigma_w, p_a, p_c, L, L_train, S, t_c_fixed, c_fixed, t_r_fixed, r_fixed):
    
    # 1. Generate anomalies and change points indexes
    
    dist_a = dist.Bernoulli(p_a)
    dist_c = dist.Bernoulli(p_c)
    
    z_a = dist_a.sample((L,))
    z_c = dist_c.sample((L,))
    
    # 1.1 Evaluation period
    
    z_a[L_train:] = 0
    z_c[L_train:] = 0
    
    z_c[t_c_fixed] = c_fixed
    # 2. Generate noises
    
    dist_e = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_e]))
    dist_o = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_o]))
    dist_u = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_u]))
    dist_r = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_r]))
    dist_v = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_v]))
    dist_w = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_w]))
    
    e = dist_e.sample((L,))
    o = dist_o.sample((L,))
    u = dist_u.sample((L,))
    r = dist_r.sample((L,))
    v = dist_v.sample((L,))
    w = dist_w.sample((L,))
    
    r[t_r_fixed] = r_fixed
    # 3. Generate alpha
    
    mu = []
    mu_with_initial_values = [torch.DoubleTensor([mu_0])]
    delta = []
    delta_with_initial_values = [torch.DoubleTensor([delta_0])]
    gamma = []
    gamma_with_initial_values = []
    
    gamma_vec = [torch.DoubleTensor(gamma_0)]
    
    
    
    for i in range(len(gamma_0)):
        gamma_with_initial_values.append(torch.DoubleTensor([gamma_0[i]]))
    
             
    for t in range(L):
        
        
        mu_t = mu_with_initial_values[-1] + delta_with_initial_values[-1]

        delta_t = delta_with_initial_values[-1] + v[t]
        

            
        
        if z_c[t] == 0:
            mu_t = mu_t + u[t]
        else:
            mu_t = mu_t + r[t]
            
        gamma_t = - sum(gamma_with_initial_values[-(S-1):]) + w[t]
        
        gamma_vec_t = torch.zeros((S-1), dtype = torch.float64)
        gamma_vec_t[0] = gamma_t
        gamma_vec_t[1:] = gamma_vec[-1][:S-2]
        
        mu.append(mu_t)
        mu_with_initial_values.append(mu_t)
        
        delta.append(delta_t)
        delta_with_initial_values.append(delta_t)
        
        gamma.append(gamma_t)
        gamma_with_initial_values.append(gamma_t)
        
        gamma_vec.append(gamma_vec_t)
            
        
        
    mu = torch.tensor(mu)
    mu_with_initial_values = torch.DoubleTensor([mu_with_initial_values])
    delta = torch.tensor(delta)
    delta_with_initial_values = torch.DoubleTensor([delta_with_initial_values])
    gamma = torch.tensor(gamma)
    gamma_with_initial_values = torch.DoubleTensor([gamma_with_initial_values])
    
    gamma_vec = gamma_vec[1:]

    # 4. Generate y
    
    y = []
    
    for t in range(L):
        
        y_t = mu[t] + gamma[t]
        
        if z_a[t] == 0:
            y_t = y_t + e[t]
        else:
            y_t = y_t + o[t]
            
        y.append(y_t)
        
    y = torch.DoubleTensor(y)
        
    return (y, mu, delta, gamma, mu_with_initial_values, delta_with_initial_values, gamma_with_initial_values, gamma_vec, z_a, z_c)
        
def generative_procedure_future(mu_0, delta_0, gamma_0, sigma_e, sigma_o, sigma_u, sigma_r, sigma_v, sigma_w, L, S):
    
   
    # 2. Generate noises
    
    dist_e = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_e]))
    dist_o = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_o]))
    dist_u = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_u]))
    dist_r = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_r]))
    dist_v = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_v]))
    dist_w = dist.Normal(torch.DoubleTensor([0.]), torch.DoubleTensor([sigma_w]))
    
    e = dist_e.sample((L,))
    o = dist_o.sample((L,))
    u = dist_u.sample((L,))
    r = dist_r.sample((L,))
    v = dist_v.sample((L,))
    w = dist_w.sample((L,))
    
    # 3. Generate alpha
    
    mu = []
    mu_with_initial_values = [torch.DoubleTensor([mu_0])]
    delta = []
    delta_with_initial_values = [torch.DoubleTensor([delta_0])]
    gamma = []
    gamma_with_initial_values = []
    
    
    
    
    for i in range(len(gamma_0)):
        gamma_with_initial_values.append(torch.DoubleTensor([gamma_0[i]]))
    
             
    for t in range(L):
        
        
        mu_t = mu_with_initial_values[-1] + delta_with_initial_values[-1]

        delta_t = delta_with_initial_values[-1] + v[t]
        

            
        
        mu_t = mu_t + u[t]

            
        gamma_t = - sum(gamma_with_initial_values[-(S-1):]) + w[t]
        

        
        mu.append(mu_t)
        mu_with_initial_values.append(mu_t)
        
        delta.append(delta_t)
        delta_with_initial_values.append(delta_t)
        
        gamma.append(gamma_t)
        gamma_with_initial_values.append(gamma_t)
        
            
        
        
    mu = torch.tensor(mu)
    mu_with_initial_values = torch.DoubleTensor([mu_with_initial_values])
    delta = torch.tensor(delta)
    delta_with_initial_values = torch.DoubleTensor([delta_with_initial_values])
    gamma = torch.tensor(gamma)
    gamma_with_initial_values = torch.DoubleTensor([gamma_with_initial_values])
    


    # 4. Generate y
    
    y = []
    
    for t in range(L):
        
        y_t = mu[t] + gamma[t]
        
        y_t = y_t + e[t]

            
        y.append(y_t)
        
    y = torch.DoubleTensor(y)
        
    return y
        
    
    
def plot_time_series(y, z_a, z_c):
    plt.figure(figsize = (20,10))
    plt.plot(np.array(y))
    plt.plot(np.array(torch.nonzero(z_a)),np.array(y[torch.nonzero(z_a)]), '*')
    for c in torch.nonzero(z_c):
        plt.axvline(x=c, linestyle='--')   
