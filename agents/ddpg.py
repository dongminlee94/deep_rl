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
   """An implementation of the DDPG agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                act_noise=0.2,
                hidden_sizes=(128,128),
                buffer_size=int(1e4),
                batch_size=64,
                actor_lr=1e-4,
                critic_lr=1e-3,
                gradient_clip_ac=0.5,
                gradient_clip_cr=1.0,
                eval_mode=False,
                actor_losses=list(),
                critic_losses=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.gamma = gamma
      self.act_noise = act_noise
      self.hidden_sizes = hidden_sizes
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.actor_lr = actor_lr
      self.critic_lr = critic_lr
      self.gradient_clip_ac = gradient_clip_ac
      self.gradient_clip_cr = gradient_clip_cr
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.critic_losses = critic_losses
      self.logger = logger

      # Main network
      self.actor = MLP(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes, output_activation=torch.tanh).to(device)
      self.critic = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(device)
      # Target network
      self.actor_target = MLP(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes, output_activation=torch.tanh).to(device)
      self.critic_target = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(device)
      
      # Initialize target parameters to match main parameters
      hard_target_update(self.actor, self.actor_target)
      hard_target_update(self.critic, self.critic_target)

      # Create optimizers
      self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
      
      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)

   def select_action(self, obs):
      action = self.actor(obs).detach().cpu().numpy()
      action += self.act_noise * np.random.randn(self.act_dim)
      return np.clip(action, -self.act_limit, self.act_limit)

   def train_model(self):
      batch = self.replay_buffer.sample(self.batch_size)
      obs1 = batch['obs1']
      obs2 = batch['obs2']
      acts = batch['acts']
      rews = batch['rews']
      done = batch['done']

      if 0: # Check shape of experiences
         print("obs1", obs1.shape)
         print("obs2", obs2.shape)
         print("acts", acts.shape)
         print("rews", rews.shape)
         print("done", done.shape)

      # Actor prediction Q(s,π(s))
      pi = self.actor(obs1)
      q_pi = self.critic(obs1, pi)
      
      # Critic prediction Q(s,a), Q‾(s',π‾(s'))
      q = self.critic(obs1, acts).squeeze(1)
      pi_target = self.actor_target(obs2)
      q_pi_target = self.critic_target(obs2, pi_target).squeeze(1)
      
      # Target for Q regression
      q_backup = rews + self.gamma*(1-done)*q_pi_target
      q_backup.to(device)

      if 0: # Check shape of prediction and target
         print("q", q.shape)
         print("q_backup", q_backup.shape)

      # DDPG losses
      actor_loss = -q_pi.mean()
      critic_loss = F.mse_loss(q, q_backup.detach())

      # Update critic network parameter
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_cr)
      self.critic_optimizer.step()
      
      # Update actor network parameter
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_ac)
      self.actor_optimizer.step()

      # Polyak averaging for target parameter
      soft_target_update(self.actor, self.actor_target)
      soft_target_update(self.critic, self.critic_target)
      
      # Save losses
      self.actor_losses.append(actor_loss.item())
      self.critic_losses.append(critic_loss.item())

   def run(self):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not done:
         self.steps += 1
         
         if self.eval_mode:
            action = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            # Collect experience (s, a, r, s') using some policy
            action = self.select_action(torch.Tensor(obs).to(device))
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Start training when the number of experience is greater than batch size
            if self.steps > self.batch_size:
               self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save logs
      self.logger['LossPi'] = round(np.mean(self.actor_losses), 5)
      self.logger['LossQ'] = round(np.mean(self.critic_losses), 5)
      return step_number, total_reward
