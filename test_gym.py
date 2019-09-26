import os
import gym
import argparse
import numpy as np
import torch
from agents.common.networks import *

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v0', 
                    help='choose an environment between CartPole-v1 and Pendulum-v0')
parser.add_argument('--algo', type=str, default='tac',
                    help='select an algorithm among dqn, ddqn, a2c, ddpg, sac, asac, tac')
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
    elif args.env == 'Pendulum-v0':
        act_dim = env.action_space.shape[0]

    if args.algo == 'dqn' or args.algo == 'ddqn':
        mlp = MLP(obs_dim, act_dim).to(device)
    elif args.algo == 'a2c':
        mlp = CategoricalPolicy(obs_dim, act_dim, activation=torch.tanh).to(device)
    elif args.algo == 'ddpg':
        mlp = MLP(obs_dim, act_dim, hidden_sizes=(128,128), output_activation=torch.tanh).to(device)
    elif args.algo == 'sac' or args.algo == 'asac' or args.algo == 'tac':
        mlp = GaussianPolicy(obs_dim, act_dim, hidden_sizes=(128,128)).to(device)

    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path)
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
            elif args.algo == 'ddpg':
                action = mlp(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            elif args.algo == 'sac' or args.algo == 'asac' or args.algo == 'tac':
                action, _, _ = mlp(torch.Tensor(obs).to(device))
                action = action.detach().cpu().numpy()
            
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
