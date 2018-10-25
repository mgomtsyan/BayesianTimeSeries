import numpy as np
import torch
from torch.distributions.normal import Normal
import pyro
import pyro.distributions as dist
from generation_with_evaluation_period import generative_procedure_with_evaluation_period


def fake_path_trick(alpha_kalman, sigma_e, sigma_o, sigma_u, sigma_r, sigma_v, sigma_w, p_a, p_c, L, L_train, S, t_c_fixed, c_fixed, t_r_fixed, r_fixed):
    
    L_train = len(alpha_kalman)
    
    mu_0_fp = 50 * np.random.rand()
    delta_0_fp = 0.

    gamma_0_fp = 0.5*np.random.rand(S-1) - 0.25
    
    (y_fp, mu_fp, delta_fp, gamma_fp, mu_with_initial_values_fp, delta_with_initial_values_fp, gamma_with_initial_values_fp, gamma_vec_fp, z_a_fp, z_c_fp) = generative_procedure_with_eval(mu_0_fp, delta_0_fp, gamma_0_fp, sigma_e, sigma_o, sigma_u, sigma_r, sigma_v, sigma_w, p_a, p_c, L_train, L_train, S, t_c_fixed, c_fixed, t_r_fixed, r_fixed)
    
    
  
    
    d = 1000.
    
    a_1 = gen_a_1(S, y)
    P_1 = gen_P_1(S, d)

    Z_t = gen_Z_t(S)
    T_t = gen_T_t(S)
    R_t = gen_R_t(S)

    H_e = sigma_e ** 2
    H_o = sigma_o ** 2

    Q_eta = gen_Q_eta(sigma_r, sigma_v, sigma_w)
    Q_xi = gen_Q_eta(sigma_u, sigma_v, sigma_w)
    
    
    a_t, P_t, K_t, F_t, v_t = kalman_filter(L_train, y_fp, a_1, P_1, Z_t, p_a, p_c, T_t, H_e, H_o, Q_eta, Q_xi, R_t)

    
    alpha_hat, V  = kalman_smoothing(L_train, v_t, a_t, F_t, K_t, P_t, T_t, Z_t)
    
    alpha_fp = []

    for i in range(len(y_fp)):
        alpha_fp_t = torch.zeros((S+1,1), dtype = torch.float64)
        alpha_fp_t[0] = mu_fp[i]
        alpha_fp_t[1] = delta_fp[i]

        alpha_fp_t[2:,0] = gamma_vec_fp[i]

        alpha_fp.append(alpha_fp_t)

    return [a_fp - a_hat + a_kalman for a_fp, a_hat , a_kalman in zip(alpha_fp,alpha_hat,alpha_kalman)]
