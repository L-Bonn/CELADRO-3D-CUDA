import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import ndimage
from itertools import product
from math import log
import math 
import sys
import matplotlib.cm
sys.path.insert(0, "celadro_3D_scripts_final/plot/")
import plot
import archive
import animation
from sklearn.metrics import mutual_info_score
from tqdm import tqdm  # if you want a progress bar


if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit(1)  
ar = archive.loadarchive(sys.argv[1])

import numpy as np
import zlib
import matplotlib.pyplot as plt

def approximate_kolmogorov_complexity(field_2d):
    """
    Approximate the Kolmogorov Complexity of a 2D field by:
      1) Flattening it into 1D.
      2) Converting to bytes (simple string encoding).
      3) Compressing (zlib).
      4) Returning the length of compressed data in bytes.
    """
    # Flatten to 1D
    flat_data = field_2d.ravel()
    # Convert numeric array to string (basic approach)
    data_str = flat_data.tobytes()  # Alternatively: str(list(flat_data)).encode('utf-8')
    
    compressed = zlib.compress(data_str, level=6)
    return len(compressed)

def measure_complexity_vs_time(fields):
    """
    For each time t, compute approximate KC of fields[t].
    fields is shape (T, Nx, Ny).

    Returns an array of shape (T,) with complexity values.
    """
    T = fields.shape[0]
    complexities = np.zeros(T, dtype=np.float64)
    for t in range(T):
        kc_t = approximate_kolmogorov_complexity(fields[t])
        complexities[t] = kc_t
    return complexities

def measure_complexity_vs_lag(fields, max_delta=10):
    """
    For each time t in [0 .. T-max_delta),
    and each delta in [1..max_delta],
    compute ratio:
        ratio(t, delta) = K(fields[t]) / K(fields[t + delta]).

    Returns:
      ratio_matrix: shape (T-max_delta, max_delta).
      valid_t: list of time indices for which we computed the ratio.
    """
    T = fields.shape[0]
    if T <= max_delta:
        raise ValueError("Not enough time steps for the specified max_delta.")

    # Precompute complexities for each time
    K_vals = [approximate_kolmogorov_complexity(fields[t]) for t in range(T)]

    ratio_matrix = np.zeros((T - max_delta, max_delta), dtype=np.float64)
    valid_t = list(range(T - max_delta))

    for i, t in enumerate(valid_t):
        Kt = K_vals[t]
        for d in range(1, max_delta+1):
            Ktpd = K_vals[t + d]
            if Ktpd == 0:
                ratio_matrix[i, d-1] = np.nan
            else:
                ratio_matrix[i, d-1] = Kt / Ktpd

    return ratio_matrix, valid_t




    	
xsec = 8
size = 1
window_size = 10;
lags = [0,1,2,5,10,20]
nbins = 16

frames = np.arange(1,253,1)
frames = np.arange(1,21,1)
cmap = mpl.cm.seismic#mpl.cm.RdBu
press_list = []
for fr in frames: 
	
	frame = ar.read_frame(fr)
	
	LX = frame.parameters['Size'][0]
	LY = frame.parameters['Size'][1]
	LZ = frame.parameters['Size'][2]
	
	sxx = frame.field_sxx
	sxx = np.reshape(sxx,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
	syy = frame.field_syy
	syy = np.reshape(syy,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
	szz = frame.field_szz
	szz = np.reshape(szz,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
	press = (1/3)*(sxx+syy+szz)
	press = press[xsec,:,:]
	press_list.append(press)
	
	del frame
	del sxx
	del syy
	del szz
	del press
	
press_fields = np.array(press_list)
print("Loaded press_fields: ",press_fields.shape)
complexities = measure_complexity_vs_time(press_fields)

# Plot
plt.figure()
plt.plot(frames, complexities, marker='o')
plt.xlabel("Time step")
plt.ylabel("Approx. KC (bytes of zlib)")
plt.title("Approx. Complexity of 2D field vs. time")
plt.show()

# 3. Measure complexity ratio vs lag
max_delta = 10
ratio_matrix, valid_t = measure_complexity_vs_lag(press_fields, max_delta=max_delta)
# ratio_matrix shape = (T-max_delta, max_delta)
# valid_t is the range of time steps for which we can compute the ratio

plt.figure()
# We'll plot ratio_matrix as an image with x-axis = t, y-axis = delta
plt.imshow(ratio_matrix.T, aspect='auto', origin='lower',
        extent=[valid_t[0], valid_t[-1], 1, max_delta])
plt.colorbar(label="K(fields[t]) / K(fields[t+delta])")
plt.xlabel("Time index t")
plt.ylabel("Time lag (delta)")
plt.title("Ratio of complexities vs. time & lag")
plt.show()

# 4. (Optional) Average ratio across time for each lag
ratio_mean = np.nanmean(ratio_matrix, axis=0)
lags = range(1, max_delta+1)
plt.figure()
plt.plot(lags, ratio_mean, marker='o')
plt.xlabel("Time lag (delta)")
plt.ylabel("Mean complexity ratio K(t)/K(t+delta)")
plt.title("Average ratio of complexities vs. lag")
plt.show()


















    

