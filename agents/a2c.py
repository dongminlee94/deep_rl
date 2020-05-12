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
                ent_coef=1e-3,
                policy_lr=1e-3,
                vf_lr=1e-3,
                eval_mode=False,
                policy_losses=list(),
                vf_losses=list(),
                entropies=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_num = act_num
      self.steps = steps 
      self.gamma = gamma
      self.ent_coef = ent_coef
      self.policy_lr = policy_lr
      self.vf_lr = vf_lr
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.vf_losses = vf_losses
      self.entropies = entropies
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
      action, log_pi, entropy, _  = self.policy(obs)
      # Prediction V(s)
      v = self.vf(obs)
      self.transition.extend([log_pi, entropy, v])
      return action.detach().cpu().numpy()

   def train_model(self):
      log_pi, entropy, v, next_obs, reward, done = self.transition

      # Prediction V(s')
      next_v = self.vf(torch.Tensor(next_obs).to(self.device))
      
      # Target for Q regression
      q = reward + self.gamma*(1-done)*next_v
      q.to(self.device)

      # Advantage = Q - V
      advant = q - v

      if 0: # Check shape of prediction and target
         print("log_pi", log_pi.shape)
         print("entropy", entropy.shape)
         print("v", v.shape)
         print("q", q.shape)

      # A2C losses
      policy_loss = -log_pi*advant.detach() + self.ent_coef*entropy
      vf_loss = F.mse_loss(v, q.detach())

      # Update value network parameter
      self.vf_optimizer.zero_grad()
      vf_loss.backward()
      self.vf_optimizer.step()
      
      # Update policy network parameter
      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()

      # Save losses & entropies
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())
      self.entropies.append(entropy.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         self.steps += 1
         
         if self.eval_mode:
            _, _, _, pi = self.policy(torch.Tensor(obs).to(self.device))
            action = pi.argmax().detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            # Create a transition list
            self.transition = []

            # Collect experience (s, a, r, s') using some policy
            action = self.select_action(torch.Tensor(obs).to(self.device))
            next_obs, reward, done, _ = self.env.step(action)

            self.transition.extend([next_obs, reward, done])
            
            self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save total average losses
      self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
      self.logger['LossV'] = round(np.mean(self.vf_losses), 5)
      self.logger['Entropy'] = round(np.mean(self.entropies), 5)
      return step_number, total_reward
