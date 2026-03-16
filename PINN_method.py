"""
Name: Project_1_PINN_Solution.py
Author: Nathan Erickson
EMail: nathan.erickson@student.nmt.edu
Description: Solves the 1D wave equation for a standing wave using a PINN

"""


#Imports
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker
import time
import torch
import torch.nn as nn


#Start timing
start = time.time()



### DEFINE NEURAL NETWROK ###

class WavePINN(nn.Module):

	"""
	This is a neural network to approximate u(x, t)

	
	Architechture:
		Input: [x, t]
		Hidden: 4 layers with 20 neurons each
		Output: u (1D)
	"""

	def __init__(self, hidden_layers = 4, neurons_per_layer=20):
		
		super(WavePINN, self).__init__()

		# Build layers list
		layers = []

		# Input layer: 2 inputs (x, t) -> neurons_per_layer
		layers.append(nn.Linear(2, neurons_per_layer))
		layers.append(nn.Tanh())

		# Hidden layers
		for _ in range(hidden_layers - 1):
			layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
			layers.append(nn.Tanh())

		# Output layer: neurons_per_layer -> 1 output (u)
		layers.append(nn.Linear(neurons_per_layer, 1))

		# Combine into sequential network
		self.network = nn.Sequential(*layers)

	def forward(self, x, t):
		"""
		Forward pass through network

		Args:
		    x: Spatial coordinate (N x 1 tensor)
		    t: Time coordinate (N x 1 tensor)

		Returns:
		    u: Temperature predictions (N x 1 tensor)
		"""
		# Concatenate x and t into single input
		inputs = torch.cat([x, t], dim=1)  # Shape: (N, 2)
		return self.network(inputs)




### FUNCTION DEFINITIONS ###


#PDE LossFunction
def compute_pde_residual(model, x, t, c):
	"""
	Compute PDE residual using automatic differentiation

	Args:
		model: Neural network
		x: Spatial points (N x 1 tensor, requires_grad=True)
		t: Time points (N x 1 tensor, requires_grad=True)
		c: wave speed

	Returns:
		residual: d^2u/dt^2 - c^2 * d^2u/dx^2 (N x 1 tensor)
	"""
	# Ensure gradients are enabled
	x.requires_grad_(True)
	t.requires_grad_(True)

	# Forward pass: compute u(x,t)
	u = model(x, t)

	# First derivatives using automatic differentiation
	u_t = torch.autograd.grad(
		outputs=u,
		inputs=t,
		grad_outputs=torch.ones_like(u),
		create_graph=True,      # Keep graph for higher derivatives
		retain_graph=True       # Don't destroy graph
	)[0]

	u_x = torch.autograd.grad(
		outputs=u,
		inputs=x,
		grad_outputs=torch.ones_like(u),
		create_graph=True,
		retain_graph=True
	)[0]

	# Second derivatives
	u_xx = torch.autograd.grad(
		outputs=u_x,
		inputs=x,
		grad_outputs=torch.ones_like(u_x),
		create_graph=True,
		retain_graph=True
	)[0]

	u_tt = torch.autograd.grad(
		outputs=u_t,
		inputs=t,
		grad_outputs=torch.ones_like(u_t),
		create_graph=True,      # Keep graph for higher derivatives
		retain_graph=True       # Don't destroy graph
	)[0]

	# Compute residual: d^u/dt^2 - c^2 * d^2u/dx^2
	residual = u_tt - c**2 * u_xx

	return residual



#Create Training Data
def create_training_data(n_domain=3800, n_boundary=400, n_initial=800):
	"""
	Create collocation points for training

	Returns:
		Dictionary with 'domain', 'boundary', 'initial' point sets
	"""
	# Domain points: random in [0,1] x [0,0.5]
	x_domain = torch.rand(n_domain, 1)
	t_domain = torch.rand(n_domain, 1) * 2

	# Boundary points: x=0 and x=1 for various t
	t_boundary = torch.rand(n_boundary, 1) * 2

	# Left boundary (x=0)
	x_boundary_left = torch.zeros(n_boundary // 2, 1)
	t_boundary_left = t_boundary[:n_boundary // 2]

	# Right boundary (x=1)
	x_boundary_right = torch.ones(n_boundary // 2, 1)
	t_boundary_right = t_boundary[n_boundary // 2:]

	# Combine boundaries
	x_boundary = torch.cat([x_boundary_left, x_boundary_right])
	t_boundary = torch.cat([t_boundary_left, t_boundary_right])

	# Initial condition points: t=0 for various x
	x_initial = torch.rand(n_initial, 1).requires_grad_(True)
	t_initial = torch.zeros(n_initial, 1).requires_grad_(True)

	# We don't need more points for the inital velocity since
	# we already have initial position points

	return {
		'domain': (x_domain, t_domain),
		'boundary': (x_boundary, t_boundary),
		'initial': (x_initial, t_initial)
	}



#Total Loss Function
def compute_loss(model, data, c):
	"""
	Compute total loss = PDE loss + BC loss + IC loss + IV Loss

	Args:
		model: Neural network
		data: Dictionary with training data
		c: wave speed

	Returns:
		Tuple of (total_loss, pde_loss, bc_loss, ic_loss, iv_loss)
	"""
	x_domain, t_domain = data['domain']
	x_boundary, t_boundary = data['boundary']
	x_initial, t_initial = data['initial']

	# 1. PDE Loss: Enforce wave away from boundaries
	residual = compute_pde_residual(model, x_domain, t_domain, c)
	loss_pde = torch.mean(residual**2)

	# 2. Boundary Condition Loss: u(0,t) = u(1,t) = 0
	u_boundary = model(x_boundary, t_boundary)
	loss_bc = torch.mean(u_boundary**2)

	# 3. Initial Displacement Loss: u(x,0) = sin(pi*x)
	u_initial = model(x_initial, t_initial)
	u_initial_true = torch.sin(np.pi * x_initial)
	loss_ic = torch.mean((u_initial - u_initial_true)**2)

	# 4. Initial Velocity Condition Loss: v(x,0) = 0
	v_init = torch.autograd.grad(u_initial, t_initial, torch.ones_like(u_initial), create_graph=True)[0]
	loss_iv = torch.mean(v_init**2)

	# Update total loss
	loss_total = loss_pde + loss_bc + loss_ic + loss_iv

	return loss_total, loss_pde, loss_bc, loss_ic, loss_iv



#Train the PINN
def train_pinn(model, data, c, epochs=5000, lr=0.001):
	"""
	Train the PINN by minimizing loss

	Args:
		model: Neural network
		data: Training data dictionary
		c: Wave speed
		epochs: Number of training iterations
		lr: Learning rate

	Returns:
	    Dictionary with loss history
	"""
	# Set up optimizer (Adam = adaptive gradient descent)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# Storage for loss history
	history = {'total': [], 'pde': [], 'bc': [], 'ic': [], 'iv': []}

	print("Starting training...")
	print(f"{'Epoch':>7} | {'Total Loss':>12} | {'PDE Loss':>12}"
			+ f" | {'BC Loss':>12} | {'IC Loss':>12}")
	print("-" * 70)

	for epoch in range(epochs):
		# Zero gradients from previous iteration
		optimizer.zero_grad()

		# Compute loss
		loss_total, loss_pde, loss_bc, loss_ic, loss_iv = compute_loss(model, data, c)

		# Backpropagation: compute gradients
		loss_total.backward()

		# Update parameters
		optimizer.step()

		# Record history
		history['total'].append(loss_total.item())
		history['pde'].append(loss_pde.item())
		history['bc'].append(loss_bc.item())
		history['ic'].append(loss_ic.item())
		history['iv'].append(loss_iv.item())

		# Print progress every 500 epochs
		if epoch % 500 == 0 or epoch == epochs - 1:
			print(f"{epoch:7d} | {loss_total.item():12.6e} | "
				  f"{loss_pde.item():12.6e} | "
				  f"{loss_bc.item():12.6e} | "
				  f"{loss_ic.item():12.6e}")

	print("\nTraining complete!")
	return history


#Function to evaluate the PINN
def evaluate_pinn(model, nx=100, nt=200):
    """
    Evaluate PINN on a regular grid for visualization

    Args:
        model: Trained neural network
        nx: Number of spatial points
        nt: Number of time points

    Returns:
        X, T, u_pred: Meshgrids and predictions
    """
    # Create test grid
    x = torch.linspace(0, 1, nx).reshape(-1, 1)
    t = torch.linspace(0, 2, nt).reshape(-1, 1)

    # Create meshgrid
    T, X = torch.meshgrid(t.squeeze(), x.squeeze(), indexing='ij')
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    # Predictions (no gradients needed)
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).reshape(X.shape)

    return X.numpy(), T.numpy(), u_pred.numpy()





### PARAMETERS ###

#Parameters
max_x = 1
max_t = 2
inv_dx = 100
inv_dt = 100
c = 1

# Space Constraints
nx = int(max_x * inv_dx)
x = np.linspace(0, max_x, nx)
dx = 1/inv_dx

#Time constraints
nt = int(max_t * inv_dt)
t = np.linspace(0, max_t, nt)
dt = 1/inv_dt


### RUN THE SIMULATION ###


# Create model
model = WavePINN(hidden_layers=4, neurons_per_layer=20)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {num_params} trainable parameters")

# Test forward pass
x_test = torch.tensor([[0.5]])  # Middle of rod
t_test = torch.tensor([[0.0]])  # At t=0
u_test = model(x_test, t_test)
print(f"Forward pass works! u(0.5, 0) = {u_test.item():.4f} (random initialization)")


# Test residual function!
x_test = torch.tensor([[0.5]], requires_grad=True)
t_test = torch.tensor([[0.1]], requires_grad=True)

residual = compute_pde_residual(model, x_test, t_test, c)
print(f"PDE residual at (x=0.5, t=0.1): {residual.item():.6f}")
print("(Should be close to 0 after training!)")

#Create training data
print("Gathering Training Data...")
data = create_training_data(n_domain=1000, n_boundary=100, n_initial=100)
print("Training Data Initialized")


# Train the model
print("Begin Training")
history = train_pinn(model, data, c, epochs=5000, lr=0.001)



# Evaluate PINN
print("Evaluating PINN on test grid...")
X, T, u_pred = evaluate_pinn(model, nx=nx, nt=nt)

# Compute analytical solution
u_true = np.zeros((nt, nx))
for i in range(nt):
	u_true[i] = np.sin(pi*x) * np.cos(c*pi*dt*i)


# Compute error
error = np.abs(u_pred - u_true)
mse = np.mean(error**2, axis=1)

print(f"Evaluation complete!")
print(f"  Grid size: {X.shape}")
print(f"  Total test points: {X.size}")



### TIME ###
end = time.time()
runtime = end - start
print(f"\nThe simulation took {runtime:.3e} seconds to run.\n")




# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Total loss (log scale)
axes[0].semilogy(history['total'], 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Total Loss (log scale)', fontsize=12)
axes[0].set_title('Total Loss During Training', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Individual components
axes[1].semilogy(history['pde'], label='PDE Loss', linewidth=2)
axes[1].semilogy(history['bc'], label='BC Loss', linewidth=2)
axes[1].semilogy(history['ic'], label='IC Loss', linewidth=2)
axes[1].semilogy(history['iv'], label='IV Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss (log scale)', fontsize=12)
axes[1].set_title('Loss Components', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()



### PLOT RESULTS ###
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,6))
fig.suptitle("Physics-Informed Neural Network Solution")

#Numerical solution
im1 = ax1.imshow(u_pred.T, extent=[0, max_t, max_x, 0], aspect='auto', cmap='RdBu')
ax1.set_title("Amplitude Over time (PINN Solution)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Space")
fig.colorbar(im1, ax=ax1, label="Amplitude")

im2 = ax2.imshow(u_true.T, extent=[0, max_t, max_x, 0], aspect='auto', cmap='RdBu')
ax2.set_title("Amplitude Over time (Analytical Solution)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Space")
fig.colorbar(im2, ax=ax2, label="Amplitude")

im3 = ax3.imshow(error.T, extent=[0, max_t, max_x, 0], aspect='auto', 
                 cmap='Reds')
ax3.set_title("Error")
ax3.set_xlabel("Time")
ax3.set_ylabel("Space")
# Add a colorbar with scientific notation for the tiny error values
cbar3 = fig.colorbar(im3, ax=ax3, label="Error Magnitude")
cbar3.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar3.ax.ticklabel_format(style='sci', scilimits=(0,0))

ax4.plot(t, mse)
ax4.set_title("Mean Squared Error")
ax4.set_ylabel("Error")
ax4.set_xlabel("Time")
ax4.set_yscale('log')

plt.tight_layout()
plt.show()