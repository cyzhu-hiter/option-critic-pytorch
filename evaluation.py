import argparse
import torch
from torch.optim import RMSprop
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from option_critic import OptionCriticFeatures, OptionCriticConv
from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import TensorboardLogger, WandbLogger, EmptyLogger
import time

# Setup argument parser
parser = argparse.ArgumentParser(description="Evaluate Option-Critic Architecture")
parser.add_argument('--env', default='CartPole-v0', help='environment to run')
parser.add_argument('--freeze-interval', type=int, default=200, help='number of steps between updates to the target network')
parser.add_argument('--cuda', action='store_true', help='use CUDA if available')
parser.add_argument('--logdir', type=str, default='runs', help='directory to save logs')
parser.add_argument('--exp', type=str, default='', help='experiment name')
parser.add_argument('--logger', choices=['none', 'tensorboard', 'wandb'], default='none', help='logger to use for tracking progress')
parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the trained model checkpoint')
parser.add_argument('--max-episode', type=int, default=100, help='identify how many episode to evaluate')
parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')


def run(args):
    # Initialize environment and models
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

    # Load model parameters
    option_critic.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    option_critic_prime = deepcopy(option_critic)
    optim = RMSprop(option_critic.parameters(), lr=args.learning_rate)

    # Set up seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)

    # Initialize logger
    logger_dict = {'none': EmptyLogger, 'tensorboard': TensorboardLogger, 'wandb': WandbLogger}
    logger = logger_dict[args.logger](logdir=args.logdir, run_name=f"{option_critic_class.__name__}-{args.env}-{args.exp}-{time.ctime()}")

    # Main evaluation loop
    ep_count = 0
    pbar = tqdm(total=args.max_history, desc="Evaluating")

    while ep_count < args.max_episode:
        obs = env.reset()
        option_lengths = {opt:[] for opt in range(args.num_options)} # TODO
        done = False; option_termination = True; ep_steps= 0

        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        env.s

        while not done and ep_steps < args.max_steps_ep:
            state = to_tensor(obs)
            action = option_critic.predict(state)
            next_obs, reward, done, _ = env.step(action)

            obs = next_obs
            ep_stepssteps += 1
            pbar.update(1)

        ep_count += 1

    pbar.close()
    try:
        logger.save()
    except:
        pass

if __name__ == "__main__":
    args = parser.parse_args('models/option_critic_fourrooms_seed=0_4op_final_params.pth')
    args.checkpoint_path = ''
    run(args)
