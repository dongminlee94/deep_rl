import os
import gym
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2', 
                    help='choose an environment between CartPole-v1 and LunarLanderContinuous-v2')
parser.add_argument('--algo', type=str, default='sac', 
                    help='select an algorithm among dqn, ddqn, a2c, ddpg, sac, asac, tac')
parser.add_argument('--training_eps', type=int, default=900, 
                    help='training episode number (CartPole: 500, LunarLanderContinuous: 900)')
parser.add_argument('--eval_per_train', type=int, default=150, 
                    help='evaluation number per training (CartPole: 50, LunarLanderContinuous: 150)')
parser.add_argument('--evaluation_eps', type=int, default=100,
                    help='evaluation episode number (CartPole: 100, LunarLanderContinuous: 100)')
parser.add_argument('--max_step', type=int, default=300,
                    help='max episode step (CartPole: 500, LunarLanderContinuous: 300)')
parser.add_argument('--threshold_return', type=int, default=190,
                    help='solved requirement for success in given environment (CartPole: 490, LunarLanderContinuous: 190)')
args = parser.parse_args()

if args.algo == 'dqn':
    from agents.dqn import Agent
elif args.algo == 'ddqn': # Just replace the target of DQN with Double DQN
    from agents.dqn import Agent
elif args.algo == 'a2c':
    from agents.a2c import Agent
elif args.algo == 'ddpg':
    from agents.ddpg import Agent
elif args.algo == 'sac':
    from agents.sac import Agent
elif args.algo == 'asac': # Automating entropy adjustment on SAC
    from agents.sac import Agent
# elif args.algo == 'tac':
#     from agents.tac import Agent

def main():
    """Main."""
    # Initialize environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    if args.env == 'CartPole-v1':
        act_dim = env.action_space.n
        act_limit = None
    elif args.env == 'LunarLanderContinuous-v2':
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    # Set a random seed
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Create an agent
    agent = Agent(env, args, obs_dim, act_dim, act_limit)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + args.env + '_' + args.algo + '_' + time.ctime() + '_alpha_' + str(0.02)
    writer = SummaryWriter(log_dir=dir_name)

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
        writer.add_scalar('Train/EpisodeReturns', train_episode_return, episode)

        # Perform the evaluation phase -- no learning
        if episode > 0 and episode % args.eval_per_train == 0:
            agent.eval_mode = True
            
            eval_sum_returns = 0.
            eval_num_episodes = 0

            for _ in range(args.evaluation_eps):
                # Run one episode
                eval_step_length, eval_episode_return = agent.run(args.max_step)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

                eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

                # Log experiment result for evaluation episodes
                writer.add_scalar('Eval/AverageReturns', eval_average_return, episode)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, episode)

            print('---------------------------------------')
            print('Episodes:', train_num_episodes)
            print('Steps:', train_step_count)
            print('AverageReturn:', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('Loss:', agent.average_losses)
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            # Save a training model
            if eval_average_return >= args.threshold_return:
                if not os.path.exists('./save_model'):
                    os.mkdir('./save_model')
                
                save_name = args.env + '_' + args.algo
                ckpt_path = os.path.join('./save_model/' + save_name + '_rt_' + str(round(train_average_return, 2)) \
                                                                    + '_ep_' + str(train_num_episodes) \
                                                                    + '_t_' + str(int(time.time() - start_time)) + '.pt')
                
                if args.algo == 'dqn' or args.algo == 'ddqn':
                    torch.save(agent.qf.state_dict(), ckpt_path)
                else:
                    torch.save(agent.actor.state_dict(), ckpt_path)
                break

if __name__ == "__main__":
    main()
