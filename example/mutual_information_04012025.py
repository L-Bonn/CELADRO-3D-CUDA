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

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit(1)  
ar = archive.loadarchive(sys.argv[1])

def discretize(data, nbins=16):
    """
    Discretize 1D continuous data into 'nbins' bins, returning
    an array of integer bin indices (0..nbins-1).
    """
    data_min, data_max = np.min(data), np.max(data)
    if data_min == data_max:
        return np.zeros_like(data, dtype=int)
    
    bin_edges = np.linspace(data_min, data_max, nbins+1)
    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins-1)
    return bin_indices
    
def mutual_info_sklearn(x, y, nbins=16):
    """
    Compute mutual information between x and y by:
      1) Discretizing x and y into nbins.
      2) Using sklearn.metrics.mutual_info_score on the discrete labels.
    """
    x_disc = discretize(x, nbins)
    y_disc = discretize(y, nbins)
    mi_value = mutual_info_score(x_disc, y_disc)
    # mutual_info_score uses natural log => MI in nats by default.
    return mi_value
    
def measure_mi_2D_sliding_window_lag(press_fields, window_size=10, lags=None, nbins=16):
    """
    Computes time-lagged MI in a sliding-window manner for 2D/3D fields.

    Args:
        press_fields: 4D np.array of shape (T, Lz, Lx, Ly), 
                      press_fields[t] is the 3D field at time t.
        window_size : how many consecutive time frames to include in each window.
        lags        : list of integer lags (>=0) to consider.
        nbins       : number of bins for discretization.

    Returns:
        mi_matrix: 2D np.array, shape = (num_windows, len(lags)),
                   where num_windows = T - window_size - max(lags).
                   mi_matrix[t_idx, lag_idx] = MI for window starting at t 
                       vs. window starting at t+lag.
        valid_t : list of valid start indices for the sliding window.
    """
    if lags is None:
        lags = [0, 1, 2, 5, 10, 20]

    T = press_fields.shape[0]
    max_lag = max(lags)
    
    # We need up to (t + window_size + max_lag) <= T
    # => t <= T - window_size - max_lag
    num_windows = T - window_size - max_lag
    if num_windows <= 0:
        raise ValueError("Not enough frames for the given window_size and max_lag.")
    
    mi_matrix = np.zeros((num_windows, len(lags)), dtype=float)
    valid_t = list(range(num_windows))

    for i, t in enumerate(valid_t):
        # Flatten the "source" window: frames [t .. t+window_size-1]
        source_block = press_fields[t : t + window_size]  # shape (window_size, Lz, Lx, Ly)
        # Flatten to 1D
        source_flat = source_block.ravel()  # shape = (window_size * Lz * Lx * Ly, )

        for j, lag in enumerate(lags):
            lag_start = t + lag
            lag_end = lag_start + window_size
            target_block = press_fields[lag_start : lag_end]
            target_flat = target_block.ravel()

            # Compute MI between these two flattened arrays
            mi_val = mutual_info_sklearn(source_flat, target_flat, nbins=nbins)
            mi_matrix[i, j] = mi_val
    
    return mi_matrix, valid_t
     
    	
xsec = 8
size = 1
window_size = 10;
lags = [0,1,2,5,10,20]
nbins = 16

frames = np.arange(1,253,1)
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
T = press_fields.shape[0]
print("Loaded press_fields: ",press_fields.shape)
mi_matrix, valid_t = measure_mi_2D_sliding_window_lag(press_fields, window_size=window_size, lags=lags, nbins=nbins)

plt.figure(figsize=(4, 4))
# Transpose so that row => lag, column => time index
# Or you can keep it as is; just be consistent with extent.
plt.imshow(mi_matrix.T, aspect='auto', origin='lower',
        extent=[valid_t[0], valid_t[-1], lags[0], lags[-1]])
plt.colorbar(label='Mutual Info (nats)')
plt.xlabel('Sliding window start index t')
plt.ylabel('Lag')
plt.title('Sliding-Window MI between full 2D fields (press)')
plt.show()

# Also, average across time for each lag
mi_mean_over_t = np.mean(mi_matrix, axis=0)
plt.figure()
plt.plot(lags, mi_mean_over_t, marker='o')
plt.xlabel('Lag')
plt.ylabel('Mean MI over time windows')
plt.title























    

