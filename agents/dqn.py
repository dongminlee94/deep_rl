import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.common.mlp import *
from agents.common.utils import *
from agents.common.buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
   """An implementation of the DQN agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                steps=0,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.995,
                target_update_step=100,
                buffer_size=int(1e5),
                batch_size=64,
                eval_mode=False,
                q_losses=list(),
                average_losses=dict()):

      self.env = env
      self.args = args
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.steps = steps 
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_decay = epsilon_decay
      self.target_update_step = target_update_step
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.eval_mode = eval_mode
      self.q_losses = q_losses
      self.average_losses = average_losses

      # Main network
      self.qf = MLP(self.obs_dim, self.act_dim).to(device)
      # Target network
      self.qf_target = MLP(self.obs_dim, self.act_dim).to(device)
      
      # Initialize target parameters to match main parameters
      hard_target_update(self.qf, self.qf_target)

      # Create optimizer
      self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=1e-3)

      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, 1, self.buffer_size)

   def select_action(self, obs):
      # Decaying epsilon
      self.epsilon *= self.epsilon_decay
      self.epsilon = max(self.epsilon, 0.01)

      if np.random.rand() <= self.epsilon:
            # Choose a random action with probability epsilon
         return np.random.randint(self.act_dim)
      else:
         # Choose the action with highest Q-value at the current state
         q_value = self.qf(obs).argmax()
         return q_value.detach().cpu().numpy()

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

      # Prediction Q(s), Q‾(s')
      q = self.qf(obs1).gather(1, acts.long()).squeeze(1)
      q_target = self.qf_target(obs2)
      
      # Target for Q regression
      if self.args.algo == 'dqn':
         q_backup = rews + self.gamma*(1-done)*q_target.max(1)[0]
         q_backup.to(device)
      else: # Double DQN
         q2 = self.qf(obs2)
         q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))
         
         q_backup = rews + self.gamma*(1-done)*q_target.max(1)[0]
         q_backup.to(device)

      if 0: # Check shape of prediction and target
         print("q", q.shape)
         print("q_backup", q_backup.shape)

      # Update perdiction network parameter
      qf_loss = F.mse_loss(q, q_backup.detach())
      self.qf_optimizer.zero_grad()
      qf_loss.backward()
      self.qf_optimizer.step()

      # Save losses
      self.q_losses.append(qf_loss)

      # Target parameters 𝜃‾ are synchronized as 𝜃 every N steps
      if self.steps % self.target_update_step == 0:
         hard_target_update(self.qf, self.qf_target)

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until we reach a terminal state.
      while not (done or step_number==max_step):
         self.steps += 1
         
         if self.eval_mode:
            q_value = self.qf(torch.Tensor(obs).to(device)).argmax()
            action = q_value.detach().cpu().numpy()
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
      
      # Save total average losses
      self.average_losses['LossQ'] = round(torch.Tensor(self.q_losses).mean().item(), 10)
      return step_number, total_reward
