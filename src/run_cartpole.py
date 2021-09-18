import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in CartPole environment')
parser.add_argument('--env', type=str, default='CartPole-v1', 
                    help='cartpole environment')
parser.add_argument('--algo', type=str, default='dqn', 
                    help='select an algorithm among dqn, ddqn, a2c')
parser.add_argument('--phase', type=str, default='train',
                    help='choose between training phase and testing phase')
parser.add_argument('--render', action='store_true', default=False,
                    help='if you want to render, set this to True')
parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=500, 
                    help='iterations to run and train agent')
parser.add_argument('--eval_per_train', type=int, default=50, 
                    help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=500,
                    help='max episode step')
parser.add_argument('--threshold_return', type=int, default=500,
                    help='solved requirement for success in given environment')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'dqn':
    from agents.dqn import Agent
elif args.algo == 'ddqn': # Just replace the target of DQN with Double DQN
    from agents.dqn import Agent
elif args.algo == 'a2c':
    from agents.a2c import Agent


def main():
    """Main."""
    # Initialize environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n
    
    print('---------------------------------------')
    print('Environment:', args.env)
    print('Algorithm:', args.algo)
    print('State dimension:', obs_dim)
    print('Action number:', act_num)
    print('---------------------------------------')

    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create an agent
    agent = Agent(env, args, device, obs_dim, act_num)

    # If we have a saved model, load it
    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        if args.algo == 'dqn' or args.algo == 'ddqn':
            agent.qf.load_state_dict(pretrained_model)
        else:
            agent.policy.load_state_dict(pretrained_model)

    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard and args.load is None:
        dir_name = 'runs/' + args.env + '/' \
                           + args.algo \
                           + '_s_' + str(args.seed) \
                           + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Main loop
    for i in range(args.iterations):
        # Perform the training phase, during which the agent learns
        if args.phase == 'train':
            agent.eval_mode = False
        
            # Run one episode
            train_step_length, train_episode_return = agent.run(args.max_step)
            
            train_num_steps += train_step_length
            train_sum_returns += train_episode_return
            train_num_episodes += 1

            train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

            # Log experiment result for training episodes
            if args.tensorboard and args.load is None:
                writer.add_scalar('Train/AverageReturns', train_average_return, i)
                writer.add_scalar('Train/EpisodeReturns', train_episode_return, i)

        # Perform the evaluation phase -- no learning
        if (i + 1) % args.eval_per_train == 0:
            eval_sum_returns = 0.
            eval_num_episodes = 0
            agent.eval_mode = True
            
            for _ in range(100):
                # Run one episode
                eval_step_length, eval_episode_return = agent.run(args.max_step)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

            eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

            # Log experiment result for evaluation episodes
            if args.tensorboard and args.load is None:
                writer.add_scalar('Eval/AverageReturns', eval_average_return, i)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, i)

            if args.phase == 'train':
                print('---------------------------------------')
                print('Iterations:', i + 1)
                print('Steps:', train_num_steps)
                print('Episodes:', train_num_episodes)
                print('EpisodeReturn:', round(train_episode_return, 2))
                print('AverageReturn:', round(train_average_return, 2))
                print('EvalEpisodes:', eval_num_episodes)
                print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('EvalAverageReturn:', round(eval_average_return, 2))
                print('OtherLogs:', agent.logger)
                print('Time:', int(time.time() - start_time))
                print('---------------------------------------')

                # Save the trained model
                if eval_average_return >= args.threshold_return:
                    if not os.path.exists('./save_model'):
                        os.mkdir('./save_model')
                    
                    ckpt_path = os.path.join('./save_model/' + args.env + '_' + args.algo \
                                                                        + '_s_' + str(args.seed) \
                                                                        + '_i_' + str(i + 1) \
                                                                        + '_tr_' + str(round(train_episode_return, 2)) \
                                                                        + '_er_' + str(round(eval_episode_return, 2)) + '.pt')
                    
                    if args.algo == 'dqn' or args.algo == 'ddqn':
                        torch.save(agent.qf.state_dict(), ckpt_path)
                    else:
                        torch.save(agent.policy.state_dict(), ckpt_path)
            elif args.phase == 'test':
                print('---------------------------------------')
                print('EvalEpisodes:', eval_num_episodes)
                print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('EvalAverageReturn:', round(eval_average_return, 2))
                print('Time:', int(time.time() - start_time))
                print('---------------------------------------')

if __name__ == "__main__":
    main()
