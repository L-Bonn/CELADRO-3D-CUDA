import numpy as np
import matplotlib.pyplot as plt

def update_ou(tcurrent, tmean, tcorr, sigma, dt):
    """
    Updates the OU process for a single step.

    tcurrent: current value of the process
    tmean: mean (long-term mean) of the process
    tcorr: correlation timescale (relaxation time)
    sigma: volatility term
    dt: discrete timestep
    """
    # Sample from a normal distribution
    dW = np.sqrt(dt) * np.random.normal(0, 1)
    # Ornstein-Uhlenbeck update
    return tcurrent - ((tcurrent - tmean) / tcorr) * dt + sigma * dW

# Simulation parameters
t0 = 1000.0         # initial value
tmean = 7520      # mean
tcorr = 5    # correlation/relaxation time
sigma = 2      # volatility
dt = 1.0         # discrete time step
n_steps = 50000    # number of time steps

params = [1,2,3]
#params = [200,500,1000]
#params = [10,80,160]
for param in params: 

	# Arrays to store simulation
	t_values = np.zeros(n_steps + 1)
	t_values[0] = t0
	# Run the simulation
	for i in range(n_steps):
	    t_values[i + 1] = update_ou(t_values[i], tmean, tcorr, param, dt)

	# Plot the result
	plt.plot(t_values,label=str(param))
	plt.xlabel('Time Step')
	plt.ylabel('OU Process Value')
	plt.title('Ornstein-Uhlenbeck Process Simulation')
	plt.legend()
plt.show()


