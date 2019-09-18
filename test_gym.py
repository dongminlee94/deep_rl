import os
import gym
import argparse
import numpy as np
import torch
from agents.common.mlp import *

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--algo', type=str, default='dqn')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--render', action="store_true", default=True)
parser.add_argument('--test_eps', type=int, default=10000)
parser.add_argument('--max_step', type=int, default=500)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    """Main."""
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n
    print('State dimension:', obs_dim)
    print('Action numbers:', act_num)

    if args.algo == 'dqn' or args.algo == 'ddqn':
        mlp = MLP(obs_dim, act_num)
    else:
        mlp = CategoricalDist(obs_dim, act_num)

    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path)
        mlp.load_state_dict(pretrained_model)

    test_sum_returns = 0.
    test_num_episodes = 0

    for episode in range(1, args.test_eps+1):
        step_number = 0
        total_reward = 0.

        obs = env.reset()
        done = False

        while not (done or step_number==args.max_step):
            if args.render:
                env.render()
            
            if args.algo == 'dqn' or args.algo == 'ddqn':
                action = mlp(torch.Tensor(obs).to(device)).argmax()
                action = action.detach().cpu().numpy()
            else:
                _, _, _, pi = mlp(torch.Tensor(obs).to(device))
                action = pi.detach().cpu().numpy().argmax()
            next_obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            step_number += 1
            obs = next_obs
        
        test_sum_returns += total_reward
        test_num_episodes += 1

        test_average_return = test_sum_returns / test_num_episodes if test_num_episodes > 0 else 0.0

        if episode % 10 == 0:
            print('---------------------------------------')
            print('Episodes:', test_num_episodes)
            print('TestAverageReturn:', test_average_return)
            print('---------------------------------------')

if __name__ == "__main__":
    main()
