import numpy as np
import argparse
import torch
from tqdm import tqdm
from copy import deepcopy

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import TensorboardLogger, WandbLogger, EmptyLogger

import time

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--lr-type',type=str, default='fixed', choices= ['fixed', 'linear', 'exponential'],
                    help='Update learning rate as training goes on.')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_eps_total', type=int, default=int(2000), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')
parser.add_argument('--checkpoint', type=bool, default=True,
                    help='whether save intermediate parameter for the network during training')
parser.add_argument('--logger',type=str,choices=['none', 'tensorboard', 'wandb'], default='wandb',
                    help='choose logger to record the progree of training')

def run(args):
    env, is_atari = make_env(args.env)
    option_critic_class = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    option_critic = option_critic_class(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    # optimizer definition
    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    if args.lr_type == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    elif args.lr_type == 'linear':
        initial_lr = args.learning_rate
        final_lr = 0.0001  # Final learning rate at the end of training
        linear_step_lr = (initial_lr - final_lr) / args.max_eps_total
    else:
        lr_scheduler = None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)

    if args.logger == 'tensorboard':
        logger = TensorboardLogger(logdir=args.logdir,
                               run_name=f"{option_critic_class.__name__}-{args.env}-{args.exp}-{time.ctime()}")
    elif args.logger == 'wandb':
        logger = WandbLogger(logdir=args.logdir, 
                             run_name=f"{option_critic_class.__name__}-{args.env}-{args.exp}-{time.ctime()}")
    else:
        logger = EmptyLogger()

    episodes = 0
    if args.switch_goal: print(f"Current goal {env.goal}")

    pbar = tqdm(total=args.max_eps_total, desc="Training Progress")

    while episodes < args.max_eps_total:
        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}
        obs   = env.reset()
        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        if logger.n_eps % 1000 == 0 and logger.n_eps > 0:
            if args.switch_goal:
                env.switch_goal()
                logger.log_customed_output(f"New goal {env.goal}")
            if args.checkpoint:
                torch.save(option_critic.state_dict(),
                           f'models/option_critic_{args.env}_seed={args.seed}_params_{logger.n_eps%1000}k.pth')
                
        if args.lr_type == 'linear':
            # Manually decrease the learning rate linearly
            for param_group in optim.param_groups:
                param_group['lr'] = max(param_group['lr'] - linear_step_lr, final_lr)
        elif args.lr_type == 'exponential':
            # Step the scheduler at the end of each episode or as needed
            if logger.n_eps % 200 == 0 and logger.n_eps > 0:
                lr_scheduler.step()

        done = False ; steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and steps < args.max_steps_ep:
            epsilon = option_critic.epsilon
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(state, current_option)
            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            curr_op_len += 1
            obs = next_obs

            
            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
        
        pbar.update(1)  # Update the progress bar by 1 step
        episodes += 1
        current_lr = next(iter(optim.param_groups))['lr']
        logger.log_episode(steps, rewards, option_lengths, steps, epsilon, current_lr)

    torch.save(option_critic.state_dict(), f'models/option_critic_{args.env}_seed={args.seed}_{args.exp}_final_params.pth')
    pbar.close()

if __name__=="__main__":
    args = parser.parse_args()

    # These if condition is for repository debugging use and as expected, right now, mainly for 
    # 1. fourroom env (developed by authors)
    # 2. cartpole env (gymaniusm  integrated)
    manual_debug = False
    if manual_debug:
        args.env = 'fourrooms'
        args.switch_goal = True
        args.logger = 'none'
        args.lr_type = 'linear'

    run(args)
