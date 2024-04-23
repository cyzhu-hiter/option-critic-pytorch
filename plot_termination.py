import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from fourrooms import Fourrooms
from option_critic import OptionCriticFeatures, OptionCriticConv

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
args = parser.parse_args()

# Initialize the environment
env = Fourrooms()

# Assuming 'OptionCriticFeatures' has the 'num_options' parameter
num_options = 4  # Set this to the number of options your model is trained with

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
model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=torch.device('cpu')))

# # Compute the termination probabilities for each state and option
# termination_probs = np.zeros((env.observation_space.shape[0], num_options))

# # Iterate over all possible states
# for state in range(env.observation_space.shape[0]):
#     # One-hot encode the state
#     one_hot_state = np.zeros(env.observation_space.shape)
#     one_hot_state[state] = 1
#     state_tensor = torch.FloatTensor(one_hot_state).unsqueeze(0)  # Add batch dimension
    
#     # Process state through the model to get the state representation
#     state_representation = model.get_state(state_tensor)
    
#     # Get termination probabilities from the model
#     termination_probs[state] = model.get_terminations(state_representation).squeeze().detach().numpy()

# Plot the termination probabilities
fig, axes = plt.subplots(2, int(num_options / 2), figsize=(10, 10))  # Adjust the layout based on the number of options

for idx, ax in enumerate(axes.flatten()):
    # Create an option-specific probability grid with NaN for walls
    option_prob = np.full(env.occupancy.shape, np.nan)

    for i in range(env.observation_space.shape[0]):
        one_hot_state = torch.zeros(env.observation_space.shape)
        one_hot_state[i] = 1

        cell = env.tocell[i]
        option_prob[cell] = model.terminations(model.get_state(one_hot_state))[:, idx].sigmoid()

    # Normalize the termination probabilities
    # norm = plt.Normalize(vmin=np.nanmin(option_prob), vmax=np.nanmax(option_prob))
    # print(np.nanmax(option_prob))

    # Choose a colormap with better contrast
    # 'plasma' is a good choice for representing a range of probabilities
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')

    # Plot the heatmap with the specified normalization and colormap
    cax = ax.imshow(option_prob, cmap=cmap, interpolation='none', vmin=0, vmax=1)

    # Optional: Create a colorbar for each subplot to show the scale
    fig.colorbar(cax, ax=ax)

    ax.set_title(f'Option {idx + 1}')
    ax.axis('off')  # Turn off the axis

plt.tight_layout()
plt.savefig(f'figs/{args.env}_termination_{num_options}op.png', bbox_inches='tight', pad_inches=0)
plt.show()  # Show the plot if desired
