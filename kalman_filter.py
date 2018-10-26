import numpy as np
import torch
from torch.distributions.normal import Normal
import pyro
import pyro.distributions as dist

def kalman_filter_using_p(L, y, a_1, P_1, Z_t, p_a_list, p_c_list, T_t, H_e, H_o, Q_eta, Q_xi, R_t):
    
    a_t = a_1
    P_t = P_1
    
    a = [a_t]
    P = [P_t]
    K = []
    v = []
    F = []
    
    for t in range(0, L - 1):
        
        p_a = p_a_list[t]
        p_c = p_c_list[t]
        
        v_t = y[t] - Z_t @ a_t
        F_t = Z_t @ P_t @ torch.t(Z_t) + p_a * H_o + (1 - p_a) * H_e
        
            
        P_tt = P_t - P_t @ torch.t(Z_t) * (1 / F_t.item()) @ Z_t @ P_t

        K_t = T_t @ P_t @ torch.t(Z_t) * (1 / F_t.item())
        
        
        
        a_t_new = T_t @ a_t + K_t @ v_t
        P_t_new = T_t @ P_tt @ torch.t(T_t) + R_t @ (p_c * Q_eta + (1 - p_c) * Q_xi) @ torch.t(R_t)
        
        a_t = a_t_new 
        P_t = P_t_new
        
        a.append(a_t)
        v.append(v_t)
        
        F.append(F_t)
        K.append(K_t)
        P.append(P_t)
        
    v_t = y[L-1] - Z_t @ a[L-1]    
    v.append(v_t)
    
    F_t = Z_t @ P[L-1] @ torch.t(Z_t) + p_a * H_o + (1 - p_a) * H_e
    F.append(F_t)
    K_t = T_t @ P[L-1] @ torch.t(Z_t) * (1 / F[L-1].item())
    K.append(K_t)
    
    return a, P, K, F, v

def kalman_filter(L, y, a_1, P_1, Z_t, p_a, p_c, T_t, H_e, H_o, Q_eta, Q_xi, R_t):
    
    a_t = a_1
    P_t = P_1
    
    a = [a_t]
    P = [P_t]
    K = []
    v = []
    F = []
    
    for t in range(0, L - 1):
        v_t = y[t] - Z_t @ a_t
        F_t = Z_t @ P_t @ torch.t(Z_t) + p_a * H_o + (1 - p_a) * H_e
        
            
        P_tt = P_t - P_t @ torch.t(Z_t) * (1 / F_t.item()) @ Z_t @ P_t

        K_t = T_t @ P_t @ torch.t(Z_t) * (1 / F_t.item())
        
        
        
        a_t_new = T_t @ a_t + K_t @ v_t
        P_t_new = T_t @ P_tt @ torch.t(T_t) + R_t @ (p_c * Q_eta + (1 - p_c) * Q_xi) @ torch.t(R_t)
        
        a_t = a_t_new 
        P_t = P_t_new
        
        a.append(a_t)
        v.append(v_t)
        
        F.append(F_t)
        K.append(K_t)
        P.append(P_t)
        
    v_t = y[L-1] - Z_t @ a[L-1]    
    v.append(v_t)
    
    F_t = Z_t @ P[L-1] @ torch.t(Z_t) + p_a * H_o + (1 - p_a) * H_e
    F.append(F_t)
    K_t = T_t @ P[L-1] @ torch.t(Z_t) * (1 / F[L-1].item())
    K.append(K_t)
    
    return a, P, K, F, v


def gen_a_1(S, y):
    
    m = S + 1
    a_1 = torch.DoubleTensor([0] * m)
    a_1[0] = y[0:S].mean()
    a_1 = a_1.view(m, -1)
    
    return a_1


def gen_P_1(S, d):
    
    m = S + 1
    diag_elems = torch.DoubleTensor([d] * m)
    P_1 = torch.diag(diag_elems)
    
    return P_1


def gen_Z_t(S):
    
    m = S + 1
    Z_t_1 = torch.DoubleTensor([1, 0, 1])
    Z_t_2 = torch.DoubleTensor([0] * (m - 3))
    Z_t = torch.cat((Z_t_1, Z_t_2), dim = 0)
    Z_t = Z_t.unsqueeze(0)
    
    return Z_t


def gen_T_t(S):
    
    T_mu = torch.ones([2, 2], dtype = torch.float64)
    T_mu[1][0] = 0
    
    m = S + 1
    diag_elems = torch.DoubleTensor([1] * (m - 3))
    T_gamma = torch.DoubleTensor(torch.diag(diag_elems, -1))
    T_gamma[0] = torch.DoubleTensor([-1] * (m - 2))
    
    block_1 = torch.cat((T_mu, torch.zeros((2, m-2), dtype = torch.float64)), dim = 1)
    block_2= torch.cat((torch.zeros((m - 2, 2), dtype = torch.float64), T_gamma), dim = 1)
    T_t = torch.cat((block_1, block_2), dim = 0)
    
    return T_t


def gen_R_t(S):
    
    m = S + 1
    R_mu = torch.eye(2, dtype = torch.float64)
    
    R_gamma = torch.zeros([m - 2, 1], dtype = torch.float64)
    R_gamma[0] = 1
    
    block_1 = torch.cat((R_mu, torch.zeros((2, 1), dtype = torch.float64)), dim = 1)
    block_2 = torch.cat((torch.zeros((m - 2, 2), dtype = torch.float64), R_gamma), dim = 1)
    R_t = torch.cat((block_1, block_2), dim = 0)
    
    return R_t


def gen_Q_eta(sigma_r, sigma_v, sigma_w):
    
    Q_eta = torch.diag(torch.tensor([sigma_r**2, sigma_v**2, sigma_w**2], dtype = torch.float64))
    
    return Q_eta


def gen_Q_xi(sigma_u, sigma_v, sigma_w):
    
    Q_xi = torch.diag(torch.tensor([sigma_u**2, sigma_v**2, sigma_w**2], dtype = torch.float64))
    
    return Q_xi
