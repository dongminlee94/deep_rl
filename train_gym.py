import os
import gym
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch')
parser.add_argument('--env_name', type=str, default='LunarLander-v2')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--algo', type=str, default='dqn')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='eg') # 'eg' or 'te'
parser.add_argument('--q', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--training_eps', type=int, default=1500)
parser.add_argument('--eval_per_train', type=int, default=100)
parser.add_argument('--evaluation_eps', type=int, default=100)
parser.add_argument('--max_step', type=int, default=500)
parser.add_argument('--threshold_return', type=int, default=200)
args = parser.parse_args()

if args.algo == 'dqn':
    from agents.dqn import Agent
elif args.algo == "ddqn":
    from agent.dqn import Agent

def main():
    """Main."""
    # Initialize environment
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n
    print('State dimension:', obs_dim)
    print('Action numbers:', act_num)

    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create an agent
    agent = Agent(env, args, obs_dim, act_num, entropic_index=args.q, alpha=args.alpha)

    # Create a SummaryWriter object by TensorBoard
    writer = SummaryWriter()

    start_time = time.time()

    train_step_count = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Runs a full experiment, spread over multiple training episodes
    for episode in range(1, args.training_eps+1):
        # Perform the training phase, during which the agent learns
        agent.eval_mode = False
        
        # Run one episode
        train_step_length, train_episode_return = agent.run(args.max_step)
        
        train_step_count += train_step_length
        train_sum_returns += train_episode_return
        train_num_episodes += 1

        train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

        # Log experiment result for training episodes
        writer.add_scalar('Train/AverageReturns', train_average_return, episode)

        # Perform the evaluation phase -- no learning
        if episode > 0 and episode % args.eval_per_train == 0:
            agent.eval_mode = True
            
            eval_step_count = 0
            eval_sum_returns = 0.
            eval_num_episodes = 0

            for _ in range(args.evaluation_eps):
                # Run one episode
                eval_step_length, eval_episode_return = agent.run(args.max_step)

                eval_step_count += eval_step_length
                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

                eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

                # Log experiment result for evaluation episodes
                writer.add_scalar('Eval/AverageReturns', eval_average_return, episode)

            print('---------------------------------------')
            print('Episodes:', train_num_episodes)
            print('Steps:', train_step_count)
            print('AverageReturn:', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('LossQ:', round(torch.Tensor(agent.q_losses).mean().item(), 3))
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            # Threshold return - Solved requirement for success in cartpole environment
            if eval_average_return >= args.threshold_return:
                if not os.path.exists('./save_model'):
                    os.mkdir('./save_model')
                
                save_name = args.env_name + '_' + args.algo
                ckpt_path = os.path.join('./save_model/' + save_name + '_rt_' + str(round(train_average_return, 2)) \
                                                                    + '_ep_' + str(train_num_episodes) + '.pt')
                
                if args.algo == 'dqn':
                    torch.save(agent.qf.state_dict(), ckpt_path)
                break

if __name__ == "__main__":
    main()
