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
                sample_size=2000,
                actor_lr=3e-4,
                critic_lr=1e-3,
                train_critic_iters=80,
                eval_mode=False,
                actor_losses=list(),
                critic_losses=list(),
                kls=list(),
                entropies=list(),
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
      self.actor_lr = actor_lr
      self.critic_lr = critic_lr
      self.train_critic_iters = train_critic_iters
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.critic_losses = critic_losses
      self.kls = kls
      self.entropies = entropies
      self.logger = logger

      # Main network
      self.actor = GaussianPolicy(self.obs_dim, self.act_dim).to(self.device)
      self.critic = MLP(self.obs_dim, 1, activation=torch.tanh).to(self.device)
      
      # Create optimizers
      self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.device, self.gamma, self.lam)

   def train_model(self):
      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act']
      ret = batch['ret']
      adv = batch['adv']

      if 0: # Check shape of experiences
         print("obs", obs.shape)
         print("act", act.shape)
         print("ret", ret.shape)
         print("adv", adv.shape)

      # Prediction logπ(s), V(s)
      _, _, dist_old, _ = self.actor(obs)
      log_pi_old = dist_old.log_prob(act)
      v = self.critic(obs).squeeze(1)
      
      if 0: # Check shape of prediction
         print("log_pi_old", log_pi_old.shape)
         print("v", v.shape)

      # VPG losses
      actor_loss = -(log_pi_old*adv).mean()
      critic_loss = F.mse_loss(v, ret)

      # Update critic network parameter
      for _ in range(self.train_critic_iters):
         self.critic_optimizer.zero_grad()
         critic_loss.backward(retain_graph=True)
         self.critic_optimizer.step()
      
      # Update actor network parameter
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Info (useful to watch during learning)
      _, _, dist, _ = self.actor(obs)
      log_pi = dist.log_prob(act)
      approx_kl = (log_pi_old - log_pi).mean()     # a sample estimate for KL-divergence, easy to compute
      approx_ent = dist.entropy().mean()           # a sample estimate for entropy, also easy to compute

      # Save losses
      self.actor_losses.append(actor_loss.item())
      self.critic_losses.append(critic_loss.item())
      self.kls.append(approx_kl.item())
      self.entropies.append(approx_ent.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         if self.eval_mode:
            action, _, _, _ = self.actor(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            self.steps += 1
            
            # Collect experience (s, a, r, s') using some policy
            _, _, _, action = self.actor(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to buffer
            val = self.critic(torch.Tensor(obs).to(self.device))
            self.buffer.add(obs, action, reward, done, val)
            
            # Start training when the number of experience is equal to sample size
            if self.steps == self.sample_size:
               self.buffer.finish_path()
               self.train_model()
               self.steps = 0

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save logs
      self.logger['LossPi'] = round(np.mean(self.actor_losses), 5)
      self.logger['LossV'] = round(np.mean(self.critic_losses), 5)
      self.logger['KL'] = round(np.mean(self.kls), 5)
      self.logger['Entropy'] = round(np.mean(self.entropies), 5)
      return step_number, total_reward
