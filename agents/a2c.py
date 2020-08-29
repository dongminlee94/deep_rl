import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.common.networks import *


class Agent(object):
   """An implementation of the Advantage Actor-Critic (A2C) agent."""

   def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_num,
                steps=0,
                gamma=0.99,
                policy_lr=1e-4,
                vf_lr=1e-3,
                eval_mode=False,
                policy_losses=list(),
                vf_losses=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_num = act_num
      self.steps = steps 
      self.gamma = gamma
      self.policy_lr = policy_lr
      self.vf_lr = vf_lr
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.vf_losses = vf_losses
      self.logger = logger

      # Policy network
      self.policy = CategoricalPolicy(self.obs_dim, self.act_num, activation=torch.tanh).to(self.device)
      # Value network
      self.vf = MLP(self.obs_dim, 1, activation=torch.tanh).to(self.device)
      
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)
      
   def select_action(self, obs):
      """Select an action from the set of available actions."""
      action, _, log_pi  = self.policy(obs)
      
      # Prediction V(s)
      v = self.vf(obs)

      # Add logπ(a|s), V(s) to transition list
      self.transition.extend([log_pi, v])
      return action.detach().cpu().numpy()

   def train_model(self):
      log_pi, v, reward, next_obs, done = self.transition

      # Prediction V(s')
      next_v = self.vf(torch.Tensor(next_obs).to(self.device))
      
      # Target for Q regression
      q = reward + self.gamma*(1-done)*next_v
      q.to(self.device)

      # Advantage = Q - V
      advant = q - v

      if 0: # Check shape of prediction and target
         print("q", q.shape)
         print("v", v.shape)
         print("log_pi", log_pi.shape)

      # A2C losses
      policy_loss = -log_pi*advant.detach()
      vf_loss = F.mse_loss(v, q.detach())

      # Update value network parameter
      self.vf_optimizer.zero_grad()
      vf_loss.backward()
      self.vf_optimizer.step()
      
      # Update policy network parameter
      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()

      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())

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
            _, pi, _ = self.policy(torch.Tensor(obs).to(self.device))
            action = pi.argmax().detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            self.steps += 1

            # Create a transition list
            self.transition = []

            # Collect experience (s, a, r, s') using some policy
            action = self.select_action(torch.Tensor(obs).to(self.device))
            next_obs, reward, done, _ = self.env.step(action)

            # Add (r, s') to transition list
            self.transition.extend([reward, next_obs, done])
            
            self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save total average losses
      self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
      self.logger['LossV'] = round(np.mean(self.vf_losses), 5)
      return step_number, total_reward
