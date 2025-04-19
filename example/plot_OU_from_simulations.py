import glob
import re
import numpy as np
import matplotlib.pyplot as plt

N = 10
#plt.figure()
for i in range(N):
	fname = 'cell_' + str(i) + '.dat'
	data = np.loadtxt(fname)
	# Create the plot for the current file
	if (len(data.shape) == 2):
		plt.figure()
		plt.plot(data[:,2],data[:, 4])  # Plotting column 0 (first column)
		# plt.plot(data[:,0],data[:,1]) 
		plt.xlabel("Index")
		plt.ylabel("Value")
		plt.title(f"Cell {i}: Plot of Column 1")
plt.show()

