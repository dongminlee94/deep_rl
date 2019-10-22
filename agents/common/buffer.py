import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(device),
                    done=torch.Tensor(self.done_buf[idxs]).to(device))


class Buffer(object):
    """
    A buffer for storing trajectories experienced by a agent interacting
    with the environment.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.don_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def add(self, obs, act, rew, don, val):
        assert self.ptr < self.max_size      # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.don_buf[self.ptr] = don
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self):
        previous_v = 0
        running_ret = 0
        running_adv = 0
        for t in reversed(range(len(self.rew_buf))):
            # The next two line computes rewards-to-go, to be targets for the value function
            running_ret = self.rew_buf[t] + self.gamma*(1-self.don_buf[t])*running_ret
            self.ret_buf[t] = running_ret

            # The next four lines implement GAE-Lambda advantage calculation
            running_del = self.rew_buf[t] + self.gamma*(1-self.don_buf[t])*previous_v - self.val_buf[t]
            running_adv = running_del + self.gamma*self.lam*(1-self.don_buf[t])*running_adv
            previous_v = self.val_buf[t]
            self.adv_buf[t] = running_adv
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()
        
    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        return dict(obs=torch.Tensor(self.obs_buf).to(device),
                    act=torch.Tensor(self.act_buf).to(device),
                    ret=torch.Tensor(self.ret_buf).to(device),
                    adv=torch.Tensor(self.adv_buf).to(device))
        