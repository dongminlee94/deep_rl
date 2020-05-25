import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.common.utils import *
from agents.common.buffers import *
from agents.common.networks import *


class Agent(object):
   """An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent."""

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
                epochs=10,
                sample_size=2048,
                mini_batch_size=64,
                clip_param=0.2,
                target_kl=0.01,
                policy_lr=3e-4,
                vf_lr=1e-3,
                gradient_clip=0.5,
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
      self.epochs = epochs
      self.sample_size = sample_size
      self.mini_batch_size = mini_batch_size
      self.clip_param = clip_param
      self.target_kl = target_kl
      self.policy_lr = policy_lr
      self.vf_lr = vf_lr
      self.gradient_clip = gradient_clip
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.vf_losses = vf_losses
      self.kls = kls
      self.logger = logger

      # Main network
      self.policy = GaussianPolicy(self.obs_dim, self.act_dim).to(self.device)
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
      log_pi_old = batch['log_pi'].detach()
      v_old = batch['v'].detach()

      for _ in range(self.epochs):
         for _ in range(self.sample_size // self.mini_batch_size):
            random_idxs = np.random.choice(self.sample_size, self.mini_batch_size)
            
            mini_obs = obs[random_idxs,:]
            mini_act = act[random_idxs,:]
            mini_ret = ret[random_idxs]
            mini_adv = adv[random_idxs]
            mini_log_pi_old = log_pi_old[random_idxs]
            mini_v_old = v_old[random_idxs]

            # Prediction logπ(s), V(s)
            _, _, mini_log_pi = self.policy(mini_obs, mini_act)
            mini_v = self.vf(mini_obs).squeeze(1)

            if 0: # Check shape of experiences & predictions with mini-batch size
               print("random_idxs", random_idxs.shape)
               print("mini_obs", mini_obs.shape)
               print("mini_act", mini_act.shape)
               print("mini_ret", mini_ret.shape)
               print("mini_adv", mini_adv.shape)
               print("mini_log_pi_old", mini_log_pi_old.shape)
               print('mini_log_pi', mini_log_pi.shape)
               print("mini_v_old", mini_v_old.shape)
               print('mini_v', mini_v.shape)

            # PPO losses
            ratio = torch.exp(mini_log_pi - mini_log_pi_old)
            clip_mini_adv = (torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)*mini_adv)
            policy_loss = -torch.min(ratio*mini_adv, clip_mini_adv).mean()
            
            clip_mini_v = mini_v_old + torch.clamp(mini_v-mini_v_old, -self.clip_param, self.clip_param)
            vf_loss = torch.max(F.mse_loss(mini_v, mini_ret), F.mse_loss(clip_mini_v, mini_ret)).mean()

            # Update value network parameter
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(self.vf.parameters(), self.gradient_clip)
            self.vf_optimizer.step()

            # A sample estimate for KL-divergence, easy to compute
            approx_kl = (mini_log_pi_old - mini_log_pi).mean()
            
            # Update policy network parameter
            if approx_kl <= 1.5 * self.target_kl:
               self.policy_optimizer.zero_grad()
               policy_loss.backward()
               nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
               self.policy_optimizer.step()

      # Info (useful to watch during learning)
      _, _, log_pi = self.policy(obs)
      kl = (log_pi_old - log_pi).mean()
      
      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())
      self.kls.append(kl.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         if self.eval_mode:
            action, _, _ = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            self.steps += 1
            
            # Collect experience (s, a, r, s') using some policy
            _, action, log_pi = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to buffer
            v = self.vf(torch.Tensor(obs).to(self.device))
            self.buffer.add(obs, action, reward, done, log_pi, v)
            
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
