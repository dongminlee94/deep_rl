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
   """An implementation of the TRPO (with support for NPG) agent."""

   def __init__(self,
                env,
                args,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                lam=0.97,
                delta=1e-2,
                hidden_sizes=(64,64),
                sample_size=2000,
                critic_lr=1e-3,
                train_critic_iters=80,
                backtrack_iter=10,
                backtrack_coeff=1.0,
                backtrack_alpha=0.5,
                eval_mode=False,
                actor_losses=list(),
                critic_losses=list(),
                actor_delta_losses=list(),
                critic_delta_losses=list(),
                kls=list(),
                backtrack_iters=list(),
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
      self.delta = delta
      self.hidden_sizes = hidden_sizes
      self.sample_size = sample_size
      self.critic_lr = critic_lr
      self.train_critic_iters = train_critic_iters
      self.backtrack_iter = backtrack_iter
      self.backtrack_coeff = backtrack_coeff
      self.backtrack_alpha = backtrack_alpha
      self.eval_mode = eval_mode
      self.actor_losses = actor_losses
      self.critic_losses = critic_losses
      self.actor_delta_losses = actor_delta_losses
      self.critic_delta_losses = critic_delta_losses
      self.kls = kls
      self.backtrack_iters = backtrack_iters
      self.logger = logger

      # Main network
      self.actor = GaussianPolicy(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes).to(device)
      self.old_actor = GaussianPolicy(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes).to(device)
      self.critic = MLP(self.obs_dim, 1, hidden_sizes=self.hidden_sizes, activation=torch.tanh).to(device)
      
      # Create optimizers
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.gamma, self.lam)

   def cg(self, obs, b, cg_iters=10, EPS=1e-8, residual_tol=1e-10):
      # Conjugate gradient algorithm
      # (https://en.wikipedia.org/wiki/Conjugate_gradient_method)
      x = torch.zeros(b.size())
      r = b.clone()
      p = r.clone()
      rdotr = torch.dot(r,r)

      for _ in range(cg_iters):
         Ap = self.hessian_vector_product(obs, p)
         alpha = rdotr / (torch.dot(p, Ap) + EPS)
         
         x += alpha * p
         r -= alpha * Ap
         
         new_rdotr = torch.dot(r, r)
         p = r + (new_rdotr / rdotr) * p
         rdotr = new_rdotr

         if rdotr < residual_tol:
            break
      return x

   def hessian_vector_product(self, obs, p, damping_coeff=0.1):
      p.detach()
      kl = self.gaussian_kl(old_actor=self.actor, new_actor=self.actor, obs=obs)
      kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
      kl_grad = self.flat_grad(kl_grad)

      kl_grad_p = (kl_grad * p).sum() 
      kl_hessian = torch.autograd.grad(kl_grad_p, self.actor.parameters())
      kl_hessian = self.flat_grad(kl_hessian, hessian=True)
      return kl_hessian + p * damping_coeff
   
   def gaussian_kl(self, old_actor, new_actor, obs):
      mu_old, std_old, _, _ = old_actor(obs)
      mu_old, std_old = mu_old.detach(), std_old.detach()
      mu, std, _, _ = new_actor(obs)

      # kl divergence between old policy and new policy : D( pi_old || pi_new )
      # (https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)
      kl = torch.log(std/std_old) + (std_old.pow(2)+(mu_old-mu).pow(2))/(2.0*std.pow(2)) - 0.5
      return kl.sum(-1, keepdim=True).mean()

   def flat_grad(self, grads, hessian=False):
      grad_flatten = []
      if hessian == False:
         for grad in grads:
            grad_flatten.append(grad.view(-1))
         grad_flatten = torch.cat(grad_flatten)
         return grad_flatten
      elif hessian == True:
         for grad in grads:
            grad_flatten.append(grad.contiguous().view(-1))
         grad_flatten = torch.cat(grad_flatten).data
         return grad_flatten

   def flat_params(self, model):
      params = []
      for param in model.parameters():
         params.append(param.data.view(-1))
      params_flatten = torch.cat(params)
      return params_flatten

   def update_model(self, model, new_params):
      index = 0
      for params in model.parameters():
         params_length = len(params.view(-1))
         new_param = new_params[index: index + params_length]
         new_param = new_param.view(params.size())
         params.data.copy_(new_param)
         index += params_length

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

      # Prediction logπ_old(s), logπ(s), V(s)
      _, _, dist_old, _ = self.actor(obs)
      log_pi_old = dist_old.log_prob(act)
      log_pi_old = log_pi_old.detach()
      _, _, dist, _ = self.actor(obs)
      log_pi = dist.log_prob(act)
      v = self.critic(obs).squeeze(1)
      
      if 0: # Check shape of prediction
         print("log_pi", log_pi.shape)
         print("v", v.shape)
   
      # TRPO losses
      ratio_old = torch.exp(log_pi - log_pi_old)
      actor_loss_old = (ratio_old*adv).mean()
      critic_loss_old = F.mse_loss(v, ret)
      
      # Update critic network parameter
      for _ in range(self.train_critic_iters):
         self.critic_optimizer.zero_grad()
         critic_loss_old.backward(retain_graph=True)
         self.critic_optimizer.step()
      v_new = self.critic(obs).squeeze(1)
      critic_loss = F.mse_loss(v_new, ret)

      # Symbols needed for CG solver
      gradient = torch.autograd.grad(actor_loss_old, self.actor.parameters())
      gradient = self.flat_grad(gradient)

      # Core calculations for NPG or TRPO
      search_dir = self.cg(obs, gradient.data)
      gHg = (self.hessian_vector_product(obs, search_dir) * search_dir).sum(0)
      step_size = torch.sqrt(2 * self.delta / gHg)
      old_params = self.flat_params(self.actor)
      self.update_model(self.old_actor, old_params)

      if self.args.algo == 'npg':
         params = old_params + step_size * search_dir
         self.update_model(self.actor, params)

         _, _, dist, _ = self.actor(obs)
         log_pi = dist.log_prob(act)
         ratio = torch.exp(log_pi - log_pi_old)
         actor_loss = (ratio*adv).mean()

         kl = self.gaussian_kl(new_actor=self.actor, old_actor=self.old_actor, obs=obs)
      elif self.args.algo == 'trpo':
         expected_improve = (gradient * step_size * search_dir).sum(0, keepdim=True)

         for i in range(self.backtrack_iter):
            # Backtracking line search
            # (https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) 464p.
            params = old_params + self.backtrack_coeff * step_size * search_dir
            self.update_model(self.actor, params)

            _, _, dist, _ = self.actor(obs)
            log_pi = dist.log_prob(act)
            ratio = torch.exp(log_pi - log_pi_old)
            actor_loss = (ratio*adv).mean()

            loss_improve = actor_loss - actor_loss_old
            expected_improve *= self.backtrack_coeff
            improve_condition = loss_improve / expected_improve

            kl = self.gaussian_kl(new_actor=self.actor, old_actor=self.old_actor, obs=obs)
            
            if kl < self.delta and improve_condition > self.backtrack_alpha:
               print('Accepting new params at step %d of line search.'%i)
               self.backtrack_iters.append(i)
               break

            if i == self.backtrack_iter-1:
               print('Line search failed! Keeping old params.')
               self.backtrack_iters.append(i)

               params = self.flat_params(self.old_actor)
               self.update_model(self.actor, params)

            self.backtrack_coeff *= 0.5

      # Save losses
      self.actor_losses.append(actor_loss_old.item())
      self.critic_losses.append(critic_loss_old.item())
      self.actor_delta_losses.append((actor_loss - actor_loss_old).item())
      self.critic_delta_losses.append((critic_loss - critic_loss_old).item())
      self.kls.append(kl.item())

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
      self.logger['DeltaLossPi'] = round(np.mean(self.actor_delta_losses), 5)
      self.logger['DeltaLossV'] = round(np.mean(self.critic_delta_losses), 5)
      self.logger['KL'] = round(np.mean(self.kls), 5)
      if self.args.algo == 'trpo':
         self.logger['BacktrackIters'] = np.mean(self.backtrack_iters)
      return step_number, total_reward
