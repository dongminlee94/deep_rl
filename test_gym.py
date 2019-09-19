import os
import gym
import argparse
import numpy as np
import torch
from agents.common.mlp import *

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2', 
                    help='choose an environment between CartPole-v1 and LunarLanderContinuous-v2')
parser.add_argument('--algo', type=str, default='ddpg',
                    help='select an algorithm among dqn, ddqn, a2c, ddpg, sac, sac_alpha, tac')
parser.add_argument('--load', type=str, default=None,
                    help='load the saved model')
parser.add_argument('--render', action="store_true", default=True,
                    help='if you want to render, set this to True')
parser.add_argument('--test_eps', type=int, default=10000,
                    help='testing episode number')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    """Main."""
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    if args.env == 'CartPole-v1':
        act_dim = env.action_space.n
    elif args.env == 'LunarLanderContinuous-v2':
        act_dim = env.action_space.shape[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    if args.algo == 'dqn' or args.algo == 'ddqn':
        mlp = MLP(obs_dim, act_dim).to(device)
    elif args.algo == 'a2c':
        mlp = CategoricalPolicy(obs_dim, act_dim).to(device)
    elif args.algo == 'ddpg':
        mlp = MLP(obs_dim, act_dim, hidden_sizes=(256,256), output_activation=torch.tanh).to(device)
    elif args.algo == 'sac' or args.algo == 'sac_alpha':
        mlp = GaussianPolicy(obs_dim, act_dim).to(device)

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

        while not done:
            if args.render:
                env.render()
            
            if args.algo == 'dqn' or args.algo == 'ddqn':
                action = mlp(torch.Tensor(obs).to(device)).argmax().detach().cpu().numpy()
            elif args.algo == 'a2c':
                _, _, _, pi = mlp(torch.Tensor(obs).to(device))
                action = pi.argmax().detach().cpu().numpy()
            elif args.algo == 'ddpg':
                action = mlp(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            elif args.algo == 'sac' or args.algo == 'sac_alpha':
                action, _, _, _ = mlp(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            # elif args.algo == 'tac':
            #     action = mlp(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            
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
