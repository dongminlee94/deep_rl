import numpy as np
import torch
from scipy.optimize import minimize

def epsilon_greedy(qf, obs, act_num, epsilon):
    if np.random.rand() <= epsilon:
        # Choose a random action with probability epsilon
        return np.random.randint(act_num)
    else:
        # Choose the action with highest Q-value at the current state
        q_value = qf(obs).argmax()
        return q_value.detach().cpu().numpy()


def tsallis_entropy(qf, obs, act_num, q, temp):
    q_values = qf(obs)
    q_logits = q_values.detach().cpu().numpy()

    _, pi = np_max_q(q_logits/temp, q=2.-q)

    pi_sum = np.sum(pi)
    if pi_sum > 1.:
        new_pi = np.zeros(pi.shape)
        for i in range(len(pi)):
          new_pi[i] = pi[i]/pi_sum
        pi = new_pi 
    return np.random.choice(act_num, p=pi.reshape(1,-1)[0])

def np_max_q(q_logits, q=1):
    maxq_list = []
    pq_list = []
    for q_logit in q_logits:
        maxq, pq = np_max_single_q(q_logit,q=q)
        pq_list.append(pq)
        maxq_list.append(maxq)
    return np.array(maxq_list), np.array(pq_list)

def np_max_single_q(q_logit, q=1.):
    q_logit = np.reshape(q_logit,[-1,])
    max_q_logit = np.max(q_logit)
    safe_q_logit = q_logit - max_q_logit
    if q==1.:
        maxq = np.log(np.sum(np.exp(safe_q_logit))) + max_q_logit
        pq = np.exp(safe_q_logit)
        pq = pq/np.sum(pq)
    else:
        obj = lambda x: -np.sum(safe_q_logit*x) - 1 / (1. - q) * (1. - np.sum(x**(2. - q)))
        const = ({'type':'eq', 'fun':lambda x:np.sum(x)-1.})
        bnds = [(0.,1.) for i in range(safe_q_logit.shape[0])]
        res = minimize(obj, x0=np.ones_like(safe_q_logit)/safe_q_logit.shape[0], constraints=const, bounds=bnds)
        maxq = -res.fun+max_q_logit
        pq = res.x
    return maxq, pq

