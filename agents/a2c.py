import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.common.networks import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
   """An implementation of the A2C agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                ent_coef=1e-3,
                actor_lr=1e-4,
                critic_lr=1e-3,
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
      self.ent_coef = ent_coef
      self.actor_lr = actor_lr
      self.critic_lr = critic_lr
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.critic_losses = critic_losses
      self.entropies = entropies
      self.logger = logger

      # Actor network
      self.actor = CategoricalPolicy(self.obs_dim, self.act_dim, activation=torch.tanh).to(device)
      # Critic network
      self.critic = MLP(self.obs_dim, 1, activation=torch.tanh).to(device)
      
      # Create optimizers
      self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
      
   def select_action(self, obs):
      """Select an action from the set of available actions."""
      action, log_pi, entropy, _  = self.actor(obs)
      # Prediction V(s)
      v = self.critic(obs)
      self.transition.extend([log_pi, entropy, v])
      return action.detach().cpu().numpy()

   def train_model(self):
      log_pi, entropy, v, next_obs, reward, done = self.transition

      # Prediction V(s')
      next_v = self.critic(torch.Tensor(next_obs).to(device))
      
      # Target for Q regression
      q = reward + self.gamma*(1-done)*next_v
      q.to(device)

      # Advantage = Q - V
      advant = q - v

      if 0: # Check shape of prediction and target
         print("log_pi", log_pi.shape)
         print("entropy", entropy.shape)
         print("v", v.shape)
         print("q", q.shape)

      # A2C losses
      actor_loss = -log_pi*advant.detach() + self.ent_coef*entropy
      critic_loss = F.mse_loss(v, q.detach())

      # Update critic network parameter
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Update actor network parameter
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Save losses & entropies
      self.actor_losses.append(actor_loss)
      self.critic_losses.append(critic_loss)
      self.entropies.append(entropy)

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         self.steps += 1
         
         if self.eval_mode:
            _, _, _, pi = self.actor(torch.Tensor(obs).to(device))
            action = pi.argmax().detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            # Create a transition
            self.transition = []

            # Collect experience (s, a, r, s') using some policy
            action = self.select_action(torch.Tensor(obs).to(device))
            next_obs, reward, done, _ = self.env.step(action)

            self.transition.extend([next_obs, reward, done])
            
            self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save total average losses
      self.logger['LossPi'] = round(torch.Tensor(self.actor_losses).to(device).mean().item(), 5)
      self.logger['LossV'] = round(torch.Tensor(self.critic_losses).to(device).mean().item(), 5)
      self.logger['Entropy'] = round(torch.Tensor(self.entropies).to(device).mean().item(), 5)
      return step_number, total_reward
