import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.common.mlp import *
from agents.common.utils import *
from agents.common.buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
   """An implementation of the SAC agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                alpha=0.05,
                automatic_entropy_tuning=False,
                buffer_size=int(1e4),
                batch_size=64,
                eval_mode=False,
                actor_losses=list(),
                qf1_losses=list(),
                qf2_losses=list(),
                vf_losses=list(),
                average_losses=dict(),
   ):

      self.env = env
      self.args = args
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.gamma = gamma
      self.alpha = alpha
      self.automatic_entropy_tuning = automatic_entropy_tuning
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.qf1_losses = qf1_losses
      self.qf2_losses = qf2_losses
      self.vf_losses = vf_losses
      self.average_losses = average_losses

      # Main network
      self.actor = GaussianPolicy(self.obs_dim, self.act_dim).to(device)
      self.qf1 = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=(256,256)).to(device)
      self.qf2 = FlattenMLP(self.obs_dim+self.act_dim, 1, hidden_sizes=(256,256)).to(device)
      self.vf = MLP(self.obs_dim, 1, hidden_sizes=(256,256)).to(device)
      # Target network
      self.vf_target = MLP(self.obs_dim, 1, hidden_sizes=(256,256)).to(device)
      
      # Initialize target parameters to match main parameters
      hard_target_update(self.vf, self.vf_target)

      # Create optimizers
      self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
      self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=3e-4)
      self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=3e-4)
      self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
      
      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)

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

      # Prediction logπ(s), Q1(s,a), Q2(s,a), V(s), V‾(s')
      _, pi, log_pi = self.actor(obs1)
      q1 = self.qf1(obs1, acts).squeeze(1)
      q2 = self.qf2(obs1, acts).squeeze(1)
      v = self.vf(obs1).squeeze(1)
      v_target = self.vf_target(obs2).squeeze(1)

      # Min Double-Q:
      min_q_pi = torch.min(self.qf1(obs1, pi), self.qf2(obs1, pi)).squeeze(1).to(device)

      # Targets for Q and V regression
      q_backup = rews + self.gamma*(1-done)*v_target
      q_backup.to(device)
      v_backup = min_q_pi - self.alpha*log_pi
      v_backup.to(device)

      if 0: # Check shape of prediction and target
         print("log_pi", log_pi.shape)
         print("q1", q1.shape)
         print("q2", q2.shape)
         print("v", v.shape)
         print("v_target", v_target.shape)
         print("min_q_pi", min_q_pi.shape)
         print("q_backup", q_backup.shape)
         print("v_backup", v_backup.shape)

      # Soft actor-critic losses
      qf1_loss = F.mse_loss(q1, q_backup.detach())
      qf2_loss = F.mse_loss(q2, q_backup.detach())
      vf_loss = F.mse_loss(v, v_backup.detach())
      actor_loss = (self.alpha*log_pi - min_q_pi).mean()

      # Update two Q network parameter
      self.qf1_optimizer.zero_grad()
      qf1_loss.backward()
      self.qf1_optimizer.step()

      self.qf2_optimizer.zero_grad()
      qf2_loss.backward()
      self.qf2_optimizer.step()
      
      # Update value network parameter
      self.vf_optimizer.zero_grad()
      vf_loss.backward()
      self.vf_optimizer.step()
      
      # Update actor network parameter
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Save loss & logpi
      self.actor_losses.append(actor_loss)
      self.qf1_losses.append(qf1_loss)
      self.qf2_losses.append(qf2_loss)
      self.vf_losses.append(vf_loss)

      # Polyak averaging for target parameter
      soft_target_update(self.vf, self.vf_target)

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until we reach a terminal state.
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

            # Add experience to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Start training when the number of experience is greater than batch size
            if self.steps > self.batch_size:
               self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save total average losses
      self.average_losses['LossPi'] = round(torch.Tensor(self.actor_losses).to(device).mean().item(), 10)
      self.average_losses['LossQ1'] = round(torch.Tensor(self.qf1_losses).to(device).mean().item(), 10)
      self.average_losses['LossQ2'] = round(torch.Tensor(self.qf2_losses).to(device).mean().item(), 10)
      self.average_losses['LossV'] = round(torch.Tensor(self.vf_losses).to(device).mean().item(), 10)
      return step_number, total_reward
