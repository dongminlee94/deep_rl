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
   """An implementation of the PPO (by clipping) agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                lam=0.97,
                hidden_sizes=(128,128),
                sample_size=2000,
                mini_batch_size=64,
                actor_lr=1e-3,
                critic_lr=1e-3,
                clip_param=0.2,
                epoch=10,
                eval_mode=False,
                actor_losses=list(),
                critic_losses=list(),
                kls=list(),
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
      self.lam = lam
      self.hidden_sizes = hidden_sizes
      self.sample_size = sample_size
      self.mini_batch_size = mini_batch_size
      self.actor_lr = actor_lr
      self.critic_lr = critic_lr
      self.clip_param = clip_param
      self.epoch = epoch
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.critic_losses = critic_losses
      self.kls = kls
      self.entropies = entropies
      self.logger = logger

      # Main network
      self.actor = GaussianPolicy(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes).to(device)
      self.critic = MLP(self.obs_dim, 1, hidden_sizes=self.hidden_sizes, activation=torch.tanh).to(device)
      
      # Create optimizers
      self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.gamma, self.lam)

   def train_model(self):
      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act']
      ret = batch['ret']
      adv = batch['adv']

      # Prediction logπ_old(s), logπ(s), V(s)
      _, _, dist_old, _ = self.actor(obs)
      log_pi_old = dist_old.log_prob(act).squeeze(1)
      log_pi_old = log_pi_old.detach()
      v_old = self.critic(obs).squeeze(1)
      v_old = v_old.detach()

      if 0: # Check shape of experiences & predictions
         print("obs", obs.shape)
         print("act", act.shape)
         print("ret", ret.shape)
         print("adv", adv.shape)
         print("log_pi_old", log_pi_old.shape)
         print("v_old", v_old.shape)

      for _ in range(self.epoch):
         for _ in range(self.sample_size // self.mini_batch_size):
            random_idxs = np.random.choice(self.sample_size, self.mini_batch_size)
            
            mini_obs = obs[random_idxs,:]
            mini_act = act[random_idxs,:]
            mini_ret = ret[random_idxs]
            mini_adv = adv[random_idxs]
            mini_log_pi_old = log_pi_old[random_idxs]
            mini_v_old = v_old[random_idxs]

            if 0: # Check shape of experiences & predictions with mini-batch size
               print("random_idxs", random_idxs.shape)
               print("mini_obs", mini_obs.shape)
               print("mini_act", mini_act.shape)
               print("mini_ret", mini_ret.shape)
               print("mini_adv", mini_adv.shape)
               print("mini_log_pi_old", mini_log_pi_old.shape)
               print("mini_v_old", mini_v_old.shape)

            _, _, dist, _ = self.actor(mini_obs)
            mini_log_pi = dist.log_prob(mini_act).squeeze(1)
            mini_v = self.critic(mini_obs).squeeze(1)

            # PPO losses
            ratio = torch.exp(mini_log_pi - mini_log_pi_old)
            clipped_ratio = (
               torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)*mini_adv
            )
            actor_loss = -torch.min(ratio*mini_adv, clipped_ratio).mean()
            clipped_value = mini_v_old + torch.clamp(
               mini_v-mini_v_old, -self.clip_param, self.clip_param
            )
            critic_loss = torch.max(F.mse_loss(mini_v, mini_ret), F.mse_loss(clipped_value, mini_ret))
            total_loss = actor_loss + 0.5 * critic_loss

            # Update critic network parameter
            self.critic_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # Update actor network parameter
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()

      # Info (useful to watch during learning)
      _, _, dist, _ = self.actor(obs)
      log_pi = dist.log_prob(act).squeeze(1)
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
            action, _, _, _ = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
         else:
            # Collect experience (s, a, r, s') using some policy
            _, _, _, action = self.actor(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
            
            self.steps += 1

            # Add experience to buffer
            val = self.critic(torch.Tensor(obs).to(device))
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
