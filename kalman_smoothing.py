import numpy as np
import torch
from torch.distributions.normal import Normal
import pyro
import pyro.distributions as dist

def kalman_smoothing(L_train, v, a, F, K, P, T_t, Z_t):
    
    alpha_hat = []
    V = []
    
      
    
    r_t = torch.zeros(T_t.shape[0], dtype = torch.float64)
    r_t = r_t.view(r_t.shape[0], -1)
    N_t = torch.zeros(T_t.shape[0], dtype = torch.float64)
    
    for i in range(L_train - 1, -1, -1):

        v_t = v[i]
        a_t = a[i]
        F_t = F[i]
        K_t = K[i]
        P_t = P[i]
        
        L_t = T_t - K_t @ Z_t
        
        
        
        
        r_prev = torch.t(Z_t) @ torch.inverse(F_t) @ v_t + torch.t(L_t) @ r_t
        
        alpha_hat_t = a_t + P_t @ r_prev
        
        
        N_prev = torch.t(Z_t) @ torch.inverse(F_t) @ Z_t +  torch.t(L_t) @ N_t @ L_t
        
        V_t = P_t -  P_t @ N_prev @ P_t
        
        alpha_hat.append(alpha_hat_t)
        V.append(V_t)
        
    alpha_hat.reverse()
    V.reverse()
            
    return alpha_hat, V
