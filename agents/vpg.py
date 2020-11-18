import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.common.utils import *
from agents.common.buffers import *
from agents.common.networks import *


class Agent(object):
   """
   An implementation of the Vanilla Policy Gradient (VPG) agent
   with GAE-Lambda for advantage estimation.
   """

   def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                lam=0.97,
                hidden_sizes=(64,64),
                sample_size=2048,
                policy_lr=1e-3,
                vf_lr=1e-3,
                train_vf_iters=80,
                eval_mode=False,
                policy_losses=list(),
                vf_losses=list(),
                kls=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.gamma = gamma
      self.lam = lam
      self.hidden_sizes = hidden_sizes
      self.sample_size = sample_size
      self.policy_lr = policy_lr
      self.vf_lr = vf_lr
      self.train_vf_iters = train_vf_iters
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.vf_losses = vf_losses
      self.kls = kls
      self.logger = logger

      # Main network
      self.policy = GaussianPolicy(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
      self.vf = MLP(self.obs_dim, 1, activation=torch.tanh).to(self.device)
      
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.device, self.gamma, self.lam)

   def train_model(self):
      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act'].detach()
      ret = batch['ret']
      adv = batch['adv']
      
      if 0: # Check shape of experiences
         print("obs", obs.shape)
         print("act", act.shape)
         print("ret", ret.shape)
         print("adv", adv.shape)

      # Update value network parameter
      for _ in range(self.train_vf_iters):
         # Prediction V(s)
         v = self.vf(obs).squeeze(1)
         
         # Value loss
         vf_loss = F.mse_loss(v, ret)

         self.vf_optimizer.zero_grad()
         vf_loss.backward()
         self.vf_optimizer.step()
      
      # Prediction logπ(s)
      _, _, _, log_pi_old = self.policy(obs, act, use_pi=False)
      log_pi_old = log_pi_old.detach()
      _, _, _, log_pi = self.policy(obs, act, use_pi=False)
      
      # Policy loss
      policy_loss = -(log_pi*adv).mean()

      # Update policy network parameter
      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()
      
      # A sample estimate for KL-divergence, easy to compute
      approx_kl = (log_pi_old - log_pi).mean()     

      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())
      self.kls.append(approx_kl.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         if self.args.render:
            self.env.render()    
         
         if self.eval_mode:
            action, _, _, _ = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            self.steps += 1
            
            # Collect experience (s, a, r, s') using some policy
            _, _, action, log_pi = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to buffer
            v = self.vf(torch.Tensor(obs).to(self.device))
            self.buffer.add(obs, action, reward, done, v)
            
            # Start training when the number of experience is equal to sample size
            if self.steps == self.sample_size:
               self.buffer.finish_path()
               self.train_model()
               self.steps = 0

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save logs
      self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
      self.logger['LossV'] = round(np.mean(self.vf_losses), 5)
      self.logger['KL'] = round(np.mean(self.kls), 5)
      return step_number, total_reward
