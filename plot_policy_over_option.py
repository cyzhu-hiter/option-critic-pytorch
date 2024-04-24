import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import make_env, to_tensor
from option_critic import OptionCriticFeatures, OptionCriticConv
import matplotlib.colors as mcolors
from torch.nn.functional import softmax

def log_norm(data):
    # Use a small offset to avoid taking the log of zero
    data_offset = data + 1e-6
    log_data = np.log(data_offset)
    return (log_data - np.nanmin(log_data)) / (np.nanmax(log_data) - np.nanmin(log_data))

# Assuming args have been parsed as shown in the previous script
parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='fourrooms', help='ROM to run')
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
parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create.'))
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
args = parser.parse_args()

# Initialize the environment
env, is_atari = make_env(args.env)

# Assuming 'OptionCriticFeatures' has the 'num_options' parameter
num_options = args.num_options

# Initialize the model
model = OptionCriticFeatures(
    in_features=env.observation_space.shape[0],
    num_actions=env.action_space.n,
    num_options=num_options,
    temperature=args.temp,
    eps_start=args.epsilon_start,
    eps_min=args.epsilon_min,
    eps_decay=args.epsilon_decay,
    eps_test=args.optimal_eps,
    device='cpu'
)

# Load your trained model weights
model_name = f'option_critic_fourrooms_seed=0_{num_options}-op-linear-lr_final_params'
try:
    model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=torch.device('cpu')))
except:
    ValueError("There does not exist the corresponding model parameters.")

# Compute the policy-over-options probabilities for each state and option
policy_probs = np.zeros((env.observation_space.shape[0], num_options))

# Iterate over all possible states
for state in range(env.observation_space.shape[0]):
    # One-hot encode the state
    one_hot_state = np.zeros(env.observation_space.shape)
    one_hot_state[state] = 1
    state_tensor = model.get_state(to_tensor(one_hot_state))  # Add batch dimension

    # Process state through the model to get the policy-over-options
    q_values = model.get_Q(state_tensor)
    # policy_probs[state] = q_values.squeeze().detach().numpy()
    policy_probs[state] = softmax(q_values, dim=-1).squeeze().detach().numpy()

nrows = 2
ncols = (num_options + 1) // 2  # Ensures enough columns for odd numbers of options

# Compute the min and max policy probability excluding NaN
min_prob = np.nanmin(policy_probs)
max_prob = np.nanmax(policy_probs)

# Define a colormap
cmap = plt.cm.viridis
cmap.set_bad(color='black')  # Color the walls in black

# Plot the policy-over-options probabilities
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10)) # Adjust based on the number of options

if num_options == 2:
    axes = np.array([axes])

for option in range(num_options):
    ax = axes.flatten()[option]  # Flatten the axes array and index linearly
    policy_grid = np.full(env.occupancy.shape, np.nan)  # Initialize with NaN for the walls

    for state in range(env.observation_space.shape[0]):
        cell = env.tocell[state]
        policy_grid[cell] = policy_probs[state, option]

    # Apply log normalization
    log_normalized_policy_grid = log_norm(policy_grid)

    # Plot the heatmap
    cax = ax.imshow(log_normalized_policy_grid, cmap=cmap, interpolation='none', norm=mcolors.Normalize(vmin=0, vmax=1))
    ax.set_title(f'Option {option + 1}')
    ax.axis('off')  # Turn off the axis

# Add a common colorbar for all subplots
cbar = fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('Log Scaled Policy Probability')

# Set the main title
fig.suptitle('Policy Over Options Probabilities')

# plt.tight_layout()
plt.savefig(f'figs/{args.env}_policy_over_options_{num_options}op.png', bbox_inches='tight', pad_inches=0)
# plt.show()
