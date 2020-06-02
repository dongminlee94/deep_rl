import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='Pendulum-v0', 
                    help='pendulum environment')
parser.add_argument('--algo', type=str, default='atac', 
                    help='select an algorithm among vpg, npg, trpo, ppo, ddpg, td3, sac, asac, tac, atac')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--training_eps', type=int, default=1000, 
                    help='training episode number')
parser.add_argument('--eval_per_train', type=int, default=50, 
                    help='evaluation number per training')
parser.add_argument('--evaluation_eps', type=int, default=100,
                    help='evaluation episode number')
parser.add_argument('--max_step', type=int, default=200,
                    help='max episode step')
parser.add_argument('--threshold_return', type=int, default=-2,
                    help='solved requirement for success in given environment')
parser.add_argument('--tensorboard', type=bool, default=True)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

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
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit)
    elif args.algo == 'sac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      alpha=0.7)
    elif args.algo == 'asac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      automatic_entropy_tuning=True)
    elif args.algo == 'tac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      alpha=0.7,
                      log_type='log-q', 
                      entropic_index=1.5)
    elif args.algo == 'atac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      log_type='log-q', 
                      entropic_index=1.5, 
                      automatic_entropy_tuning=True)
    else: # vpg, npg, trpo, ppo
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit)

    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard:
        dir_name = 'runs/' + args.env + '/' \
                           + args.algo \
                           + '_s_' + str(args.seed) \
                           + '_a_0.7_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Runs a full experiment, spread over multiple training episodes
    for episode in range(1, args.training_eps+1):
        # Perform the training phase, during which the agent learns
        agent.eval_mode = False
        
        # Run one episode
        train_step_length, train_episode_return = agent.run(args.max_step)
        
        train_num_steps += train_step_length
        train_sum_returns += train_episode_return
        train_num_episodes += 1

        train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

        # Log experiment result for training episodes
        if args.tensorboard:
            writer.add_scalar('Train/AverageReturns', train_average_return, episode)
            writer.add_scalar('Train/EpisodeReturns', train_episode_return, episode)
            if args.algo == 'asac' or args.algo == 'atac':
                writer.add_scalar('Train/Alpha', agent.alpha, episode)

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
            if args.tensorboard:
                writer.add_scalar('Eval/AverageReturns', eval_average_return, episode)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, episode)

            print('---------------------------------------')
            print('Steps:', train_num_steps)
            print('Episodes:', train_num_episodes)
            print('AverageReturn:', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('OtherLogs:', agent.logger)
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            # Save the trained model
            if eval_average_return >= args.threshold_return:
                if not os.path.exists('./tests/save_model'):
                    os.mkdir('./tests/save_model')
                
                ckpt_path = os.path.join('./tests/save_model/' + args.env + '_' + args.algo \
                                                                          + '_s_' + str(args.seed) \
                                                                          + '_ep_' + str(train_num_episodes) \
                                                                          + '_tr_' + str(round(train_average_return, 2)) \
                                                                          + '_er_' + str(round(eval_average_return, 2)) + '.pt')
                
                torch.save(agent.policy.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()
