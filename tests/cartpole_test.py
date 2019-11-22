import os
import gym
import argparse
import numpy as np
import torch
from common.networks import *

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='dqn',
                    help='select an algorithm among dqn, ddqn, a2c')
parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it (ex. --load=CartPole-v1_...)')
parser.add_argument('--render', action="store_true", default=True,
                    help='if you want to render, set this to True')
parser.add_argument('--test_eps', type=int, default=10000,
                    help='testing episode number')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    """Main."""
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n

    if args.algo == 'dqn' or args.algo == 'ddqn':
        mlp = MLP(obs_dim, act_num).to(device)
    elif args.algo == 'a2c':
        mlp = CategoricalPolicy(obs_dim, act_num, activation=torch.tanh).to(device)

    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        mlp.load_state_dict(pretrained_model)

    test_sum_returns = 0.
    test_num_episodes = 0

    for episode in range(1, args.test_eps+1):
        total_reward = 0.

        obs = env.reset()
        done = False

        while not done:
            if args.render:
                env.render()
            
            if args.algo == 'dqn' or args.algo == 'ddqn':
                action = mlp(torch.Tensor(obs).to(device)).argmax().detach().cpu().numpy()
            elif args.algo == 'a2c':
                _, _, _, pi = mlp(torch.Tensor(obs).to(device))
                action = pi.argmax().detach().cpu().numpy()
            
            next_obs, reward, done, _ = env.step(action)
            
            total_reward += reward
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
