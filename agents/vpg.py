import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.common.utils import *
from agents.common.buffer import *
from agents.common.networks import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
   """An implementation of the VPG agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                policy_mode='on-policy',
                hidden_sizes=(128,128),
                sample_size=2000,
                actor_lr=1e-4,
                critic_lr=3e-3,
                eval_mode=False,
                actor_losses=list(),
                critic_losses=list(),
                entropies=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.gamma = gamma
      self.policy_mode = policy_mode
      self.hidden_sizes = hidden_sizes
      self.sample_size = sample_size
      self.actor_lr = actor_lr
      self.critic_lr = critic_lr
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.critic_losses = critic_losses
      self.entropies = entropies
      self.logger = logger

      # Main network
      self.actor = GaussianPolicy(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes, 
                                 activation=torch.tanh, policy_mode=self.policy_mode).to(device)
      self.critic = MLP(self.obs_dim, 1, hidden_sizes=self.hidden_sizes, activation=torch.tanh).to(device)
      
      # Create optimizers
      self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size)

   def get_return(self, rew, don):
      ret = torch.zeros_like(rew)
      running_ret = 0

      for t in reversed(range(len(rew))):
         running_ret = rew[t] + self.gamma*(1-don[t])*running_ret
         ret[t] = running_ret
      
      ret = (ret - ret.mean()) / ret.std()
      return ret

   def train_model(self):
      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act']
      rew = batch['rew']
      don = batch['don']

      if 0: # Check shape of experiences
         print("obs", obs.shape)
         print("act", act.shape)
         print("rew", rew.shape)
         print("don", don.shape)

      # Prediction logπ(s), V(s)
      _, _, log_pi = self.actor(obs)
      v = self.critic(obs).squeeze(1)
      
      # Compute the rewards-to-go for each state
      ret = self.get_return(rew, don)
      
      # Advantage = G - V
      adv = ret - v

      if 0: # Check shape of prediction and return
         print("log_pi", log_pi.shape)
         print("v", v.shape)
         print("ret", ret.shape)

      # VPG losses
      actor_loss = -(log_pi*adv.detach()).mean()
      critic_loss = F.mse_loss(v, ret)

      # Update critic network parameter
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Update actor network parameter
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Info (useful to watch during learning)
      entropy = (-log_pi).mean()

      # Save log
      self.actor_losses.append(actor_loss)
      self.critic_losses.append(critic_loss)
      self.entropies.append(entropy)

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number==max_step):
         self.steps += 1
         
         if self.eval_mode:
            action, _, _ = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            # Collect experience (s, a, r, s') using some policy
            _, action, _ = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to buffer
            self.buffer.add(obs, action, reward, done)
            
            # Start training when the number of experience is greater than batch size
            if self.steps % self.sample_size == 0:
               self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save logs
      self.logger['LossPi'] = round(torch.Tensor(self.actor_losses).to(device).mean().item(), 10)
      self.logger['LossV'] = round(torch.Tensor(self.critic_losses).to(device).mean().item(), 10)
      self.logger['Entropy'] = round(torch.Tensor(self.entropies).to(device).mean().item(), 10)
      return step_number, total_reward
