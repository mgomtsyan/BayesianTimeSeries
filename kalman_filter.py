import numpy as np
import torch
from torch.distributions.normal import Normal
import pyro
import pyro.distributions as dist
from generation_with_evaluation_period import generative_procedure_with_evaluation_period


def kalman_filter(L, m, y, a_1, P_1, Z_t, p_a, p_c, T_t, H_e, H_o, Q_eta, Q_xi, R_t):
    
    a_t = a_1
    P_t = P_1
    
    a = [a_t]
    P = [P_t]
    
    for t in range(0, L - 1):
        v_t = y[t] + Z_t @ a_t
        F_t = Z_t @ P_t @ torch.t(Z_t) + p_a * H_o + (1 - p_a) * H_e
        
        a_tt = a_t + P_t @ torch.t(Z_t) * (1 / F_t.item()) * v_t
        P_tt = P_t - P_t @ torch.t(Z_t) * (1 / F_t.item()) @ Z_t @ P_t

        K_t = T_t @ P_t @ torch.t(Z_t) * (1 / F_t.item())
        
        a_t_new = T_t @ a_t + K_t @ v_t
        P_t_new = T_t @ P_tt @ torch.t(T_t) + R_t @ (p_c * Q_eta + (1 - p_c * Q_xi)) @ torch.t(R_t)
        
        a_t = a_t_new 
        P_t = P_t_new
        
        a.append(a_t)
        P.append(P_t)
    
    return a, P


def gen_a_1(S, y):
    
    m = S + 1
    a_1 = torch.FloatTensor([0] * m)
    a_1[0] = y[0:S].mean()
    a_1 = a_1.view(8, -1)
    
    return a_1


def gen_P_1(S, d):
    
    m = S + 1
    diag_elems = torch.FloatTensor([d] * m)
    P_1 = torch.diag(diag_elems)
    
    return P_1


def gen_Z_t(S):
    
    m = S + 1
    Z_t = torch.FloatTensor([1,0] * int(m / 2))
    Z_t = Z_t.unsqueeze(0)
    
    return Z_t


def gen_T_t(S):
    
    T_mu = torch.ones([2, 2])
    T_mu[1][0] = 0
    
    m = S + 1
    diag_elems = torch.FloatTensor([1] * (m - 3))
    T_gamma = torch.FloatTensor(torch.diag(diag_elems, -1))
    T_gamma[0] = torch.FloatTensor([-1] * (m - 2))
    
    block_1 = torch.cat((T_mu, torch.zeros((2, m-2))), dim = 1)
    block_2= torch.cat((torch.zeros((m - 2, 2)), T_gamma), dim = 1)
    T_t = torch.cat((block_1, block_2), dim = 0)
    
    return T_t


def gen_R_t(S, r):
    
    m = S + 1
    R_mu = torch.eye(2)
    
    R_gamma = torch.zeros([m - 2, 1])
    R_gamma[0] = 1
    
    block_1 = torch.cat((R_mu, torch.zeros((2, 1))), dim = 1)
    block_2 = torch.cat((torch.zeros((m - 2, 2)), R_gamma), dim = 1)
    R_t = torch.cat((block_1, block_2), dim = 0)
    
    return R_t


def gen_Q_eta(sigma_r, sigma_v, sigma_w):
    
    Q_eta = torch.diag(torch.tensor([sigma_r**2, sigma_v**2, sigma_w**2]))
    
    return Q_eta


def gen_Q_xi(sigma_u, sigma_v, sigma_w):
    
    Q_xi = torch.diag(torch.tensor([sigma_u**2, sigma_v**2, sigma_w**2]))
    
    return Q_xi
