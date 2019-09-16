import os
import gym
import argparse
import numpy as np
import torch
from agents.common.mlp import MLP

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='CartPole-v1')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--render', action="store_true", default=True)
parser.add_argument('--test_eps', type=int, default=10000)
parser.add_argument('--max_step', type=int, default=500)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    """Main."""
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n
    print('State dimension:', obs_dim)
    print('Action numbers:', act_num)

    qf = MLP(obs_dim, act_num)

    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path)
        qf.load_state_dict(pretrained_model)

    for episode in range(args.test_eps):
        step_number = 0
        total_reward = 0.

        obs = env.reset()
        done = False

        while not (done or step_number==args.max_step):
            if args.render:
                env.render()

            q_value = qf(torch.Tensor(obs).to(device)).argmax()
            action = q_value.detach().cpu().numpy()
            next_obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            step_number += 1
            obs = next_obs
                
        if episode % 10 == 0:
            print('Episodes:', episode)
            print('TestReturn:', total_reward)

if __name__ == "__main__":
    main()
