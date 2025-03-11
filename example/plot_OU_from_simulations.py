import glob
import re
import numpy as np
import matplotlib.pyplot as plt
'''
fname = 'cell_28.dat'
data = np.loadtxt(fname) 
plt.figure()
plt.plot(data[:,2])
plt.show()
'''
# Find all files matching the pattern
file_pattern = "cell_*.dat"
files = glob.glob(file_pattern)
files.sort()  # Optional: sort files for consistent ordering

#plt.figure()
for fname in files:
    # Extract the numerical part from the filename (e.g., cell_1.dat -> 1)
    match = re.search(r'cell_(\d+)\.dat', fname)
    if match:
        cell_number = match.group(1)
    else:
        cell_number = fname  # Fallback to filename if no number is found

    # Load the data (adjust method if your data is not plain text)
    data = np.loadtxt(fname)

    # Create the plot for the current file
    plt.figure()
    plt.plot(data[:,2],data[:, 4])  # Plotting column 0 (first column)
    # plt.plot(data[:,0],data[:,1]) 
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Cell {cell_number}: Plot of Column 1")
    plt.show()

