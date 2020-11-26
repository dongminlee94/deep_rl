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
   An implementation of agents for Soft Actor-Critic (SAC), SAC with automatic entropy adjustment (SAC-AEA).
   """

   def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                expl_before=2000,
                train_after=1000,
                gamma=0.99,
                alpha=0.2,
                automatic_entropy_tuning=False,
                hidden_sizes=(128,128),
                buffer_size=int(1e4),
                batch_size=64,
                policy_lr=3e-4,
                qf_lr=3e-4,
                eval_mode=False,
                policy_losses=list(),
                qf1_losses=list(),
                qf2_losses=list(),
                alpha_losses=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.expl_before = expl_before
      self.train_after = train_after
      self.gamma = gamma
      self.alpha = alpha
      self.automatic_entropy_tuning = automatic_entropy_tuning
      self.hidden_sizes = hidden_sizes
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.policy_lr = policy_lr
      self.qf_lr = qf_lr
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.qf1_losses = qf1_losses
      self.qf2_losses = qf2_losses
      self.alpha_losses = alpha_losses
      self.logger = logger

      # Main network
      self.policy = ReparamGaussianPolicy(self.obs_dim, self.act_dim, self.act_limit,
                                          hidden_sizes=self.hidden_sizes).to(self.device)
      self.qf1 = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
      self.qf2 = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
      # Target network
      self.qf1_target = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
      self.qf2_target = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
      
      # Initialize target parameters to match main parameters
      hard_target_update(self.qf1, self.qf1_target)
      hard_target_update(self.qf2, self.qf2_target)

      # Concat the Q-network parameters to use one optim
      self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.qf_optimizer = optim.Adam(self.qf_parameters, lr=self.qf_lr)
      
      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size, self.device)

      # If automatic entropy tuning is True, 
      # initialize a target entropy, a log alpha and an alpha optimizer
      if self.automatic_entropy_tuning:
         self.target_entropy = -np.prod((act_dim,)).item()
         self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
         self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.policy_lr)

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

      # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'), Q1(s,a), Q2(s,a)
      _, pi, log_pi = self.policy(obs1)
      _, next_pi, next_log_pi = self.policy(obs2)
      q1 = self.qf1(obs1, acts).squeeze(1)
      q2 = self.qf2(obs1, acts).squeeze(1)

      # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))), min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
      min_q_pi = torch.min(self.qf1(obs1, pi), self.qf2(obs1, pi)).squeeze(1).to(self.device)
      min_q_next_pi = torch.min(self.qf1_target(obs2, next_pi), 
                                self.qf2_target(obs2, next_pi)).squeeze(1).to(self.device)

      # Targets for Q regression
      v_backup = min_q_next_pi - self.alpha*next_log_pi
      q_backup = rews + self.gamma*(1-done)*v_backup
      q_backup.to(self.device)

      if 0: # Check shape of prediction and target
         print("log_pi", log_pi.shape)
         print("next_log_pi", next_log_pi.shape)
         print("q1", q1.shape)
         print("q2", q2.shape)
         print("min_q_pi", min_q_pi.shape)
         print("min_q_next_pi", min_q_next_pi.shape)
         print("q_backup", q_backup.shape)

      # SAC losses
      policy_loss = (self.alpha*log_pi - min_q_pi).mean()
      qf1_loss = F.mse_loss(q1, q_backup.detach())
      qf2_loss = F.mse_loss(q2, q_backup.detach())
      qf_loss = qf1_loss + qf2_loss

      # Update policy network parameter
      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()
      
      # Update two Q-network parameter
      self.qf_optimizer.zero_grad()
      qf_loss.backward()
      self.qf_optimizer.step()

      # If automatic entropy tuning is True, update alpha
      if self.automatic_entropy_tuning:
         alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
         self.alpha_optimizer.zero_grad()
         alpha_loss.backward()
         self.alpha_optimizer.step()

         self.alpha = self.log_alpha.exp()

         # Save alpha loss
         self.alpha_losses.append(alpha_loss.item())

      # Polyak averaging for target parameter
      soft_target_update(self.qf1, self.qf1_target)
      soft_target_update(self.qf2, self.qf2_target)
      
      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.qf1_losses.append(qf1_loss.item())
      self.qf2_losses.append(qf2_loss.item())

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
            action, _, _ = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            self.steps += 1

            # Until expl_before have elapsed, randomly sample actions 
            # from a uniform distribution for better exploration. 
            # Afterwards, use the learned policy.
            if self.steps > self.expl_before:
               _, action, _ = self.policy(torch.Tensor(obs).to(self.device))
               action = action.detach().cpu().numpy()
            else:
               action = self.env.action_space.sample()

            # Collect experience (s, a, r, s') using some policy
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Start training when the number of experience is greater than train_after
            if self.steps > self.train_after:
               self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save logs
      self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
      self.logger['LossQ1'] = round(np.mean(self.qf1_losses), 5)
      self.logger['LossQ2'] = round(np.mean(self.qf2_losses), 5)
      if self.automatic_entropy_tuning:
         self.logger['LossAlpha'] = round(np.mean(self.alpha_losses), 5)
      return step_number, total_reward
