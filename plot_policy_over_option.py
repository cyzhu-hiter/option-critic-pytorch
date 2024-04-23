import torch
import numpy as np
import matplotlib.pyplot as plt
from fourrooms import Fourrooms
from option_critic import OptionCriticFeatures

# Initialize environment and model
env = Fourrooms()
model = OptionCriticFeatures(
    in_features=np.prod(env.observation_space.shape),
    num_actions=env.action_space.n,
    num_options=4,  # Change this to the number of options you have
    device='cpu',
    testing=True
)

# Load your trained model weights
model_name = 'option_critic_fourrooms_seed=0_4-op-linear-lr_final_params'
model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=torch.device('cpu')))

# Initialize termination probability grid
termination_grid = np.full(env.occupancy.shape, np.nan)

# Compute the termination probabilities for each state and option
for state in range(env.observation_space.shape[0]):
    # One-hot encode the state
    one_hot_state = np.zeros(env.observation_space.shape)
    one_hot_state[state] = 1
    state_tensor = torch.FloatTensor(one_hot_state).unsqueeze(0)  # Add batch dimension
    
    # Get the termination probabilities from the model
    termination_prob = model.get_terminations(state_tensor).squeeze().detach().numpy()

    # Get the grid cell coordinates from the state number
    cell = env.tocell[state]

    # Update the termination probability grid
    termination_grid[cell[0], cell[1]] = termination_prob

# Plot the termination probabilities
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the layout based on the number of options

for idx, ax in enumerate(axes.flatten()):
    # Extract the option-specific termination probabilities
    option_grid = np.ma.masked_invalid(termination_grid)
    
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')  # Color the walls in black
    
    # Plot the heatmap
    ax.imshow(option_grid, cmap=cmap, interpolation='none')
    ax.set_title(f'Option {idx + 1} Termination Probabilities')
    ax.axis('off')  # Turn off the axis

plt.tight_layout()
plt.savefig('fourroom_termination.png', bbox_inches='tight', pad_inches=0)
plt.show()  # Show the plot if desired
