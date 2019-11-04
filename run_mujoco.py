import os
import gym
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in MuJoCo environments')
parser.add_argument('--env', type=str, default='HalfCheetah-v2', 
                    help='choose an environment between HalfCheetah-v2, Ant-v2, Pusher-v2 and Humanoid-v2')
parser.add_argument('--algo', type=str, default='ddpg', 
                    help='select an algorithm among vpg, npg, trpo, ppo, ddpg, td3, sac, asac, tac, atac')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=100, 
                    help='iterations to run and train agent')
parser.add_argument('--max_step', type=int, default=1000,
                    help='max episode step')
args = parser.parse_args()

if args.algo == 'vpg':
    from agents.vpg import Agent
elif args.algo == 'npg':
    from agents.trpo import Agent
elif args.algo == 'trpo':
    from agents.trpo import Agent
elif args.algo == 'ppo':
    from agents.ppo import Agent
elif args.algo == 'ddpg':
    from agents.ddpg import Agent
elif args.algo == 'td3':
    from agents.td3 import Agent
elif args.algo == 'sac':
    from agents.sac import Agent
elif args.algo == 'asac': # Automating entropy adjustment on SAC
    from agents.sac import Agent
elif args.algo == 'tac': 
    from agents.sac import Agent
elif args.algo == 'atac': # Automating entropy adjustment on TAC
    from agents.sac import Agent

def main():
    """Main."""
    # Initialize environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create an agent
    if args.algo == 'ddpg' or args.algo == 'td3':
        agent = Agent(env, args, obs_dim, act_dim, act_limit, act_noise=0.1, 
                    hidden_size=(400,300), buffer_size=int(1e6), batch_size=100)
    elif args.algo == 'sac':
        agent = Agent(env, args, obs_dim, act_dim, act_limit, alpha=0.2, 
                    hidden_size=(400,300), buffer_size=int(1e6), batch_size=100)
    elif args.algo == 'asac':
        agent = Agent(env, args, obs_dim, act_dim, act_limit, automatic_entropy_tuning=True, 
                    hidden_size=(400,300), buffer_size=int(1e6), batch_size=100)
    elif args.algo == 'tac':
        agent = Agent(env, args, obs_dim, act_dim, act_limit, alpha=0.2, 
                    log_type='log-q', entropic_index=1.5, 
                    hidden_size=(400,300), buffer_size=int(1e6), batch_size=100)
    elif args.algo == 'atac':
        agent = Agent(env, args, obs_dim, act_dim, act_limit, 
                    log_type='log-q', entropic_index=1.5, automatic_entropy_tuning=True,
                    hidden_size=(400,300), buffer_size=int(1e6), batch_size=100)
    else:
        agent = Agent(env, args, obs_dim, act_dim, act_limit, 
                    hidden_size=(400,300), buffer_size=int(1e6), batch_size=100)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + args.env + '/' + args.algo + '/' + str(args.seed) + '_' + time.ctime()
    writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_step_count = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Runs a full experiment, spread over multiple training episodes
    for i in range(args.iterations):
        # Perform the training phase, during which the agent learns
        agent.eval_mode = False
        
        # Run one episode
        train_step_length, train_episode_return = agent.run()
        
        train_step_count += train_step_length
        train_sum_returns += train_episode_return
        train_num_episodes += 1

        train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

        # Log experiment result for training episodes
        writer.add_scalar('Train/AverageReturns', train_average_return, episode)
        writer.add_scalar('Train/EpisodeReturns', train_episode_return, episode)
        if args.algo == 'asac' or args.algo == 'atac':
            writer.add_scalar('Train/Alpha', agent.alpha, episode)

        # Perform the evaluation phase -- no learning
        if (i > 0) and (i % 10 == 0):
            agent.eval_mode = True
            
            eval_sum_returns = 0.
            eval_num_episodes = 0

            for _ in range(10):
                # Run one episode
                eval_step_length, eval_episode_return = agent.run()

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
            print('OtherLogs:', agent.logger)
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            # Save a training model
            if not os.path.exists('./tests/save_model'):
                os.mkdir('./tests/save_model')
            
            save_name = args.env + '_' + args.algo
            ckpt_path = os.path.join('./tests/save_model/' + save_name + '_i_' + str(i) \
                                                                       + '_ep_' + str(train_num_episodes) \
                                                                       + '_st_' + str(train_step_count) \
                                                                       + '_rt_' + str(round(train_average_return, 2)) \
                                                                       + '_t_' + str(int(time.time() - start_time)) + '.pt')
            
            torch.save(agent.actor.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()
