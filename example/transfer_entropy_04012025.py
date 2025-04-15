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

# --- From your existing code snippets ---
def discretize(data, nbins=16):
    """
    Discretize 1D continuous data into 'nbins' bins, returning
    an array of integer bin indices (0..nbins-1).
    """
    data_min, data_max = np.min(data), np.max(data)
    # Handle constant data:
    if data_min == data_max:
        return np.zeros_like(data, dtype=int)
    
    bin_edges = np.linspace(data_min, data_max, nbins+1)
    bin_indices = np.digitize(data, bin_edges) - 1
    # Ensure it stays within [0, nbins-1]
    bin_indices = np.clip(bin_indices, 0, nbins-1)
    return bin_indices

def transfer_entropy_continuous(x, y, bins=16, lag=1, eps=1e-12):
    """
    Estimate Transfer Entropy T_{X->Y} for continuous 1D arrays x, y using
    a naive histogram-based approach with single lag (lag=1 by default).

    T_{X->Y} = sum p(x_t, y_t, y_{t+1}) * log2( p(y_{t+1}| x_t, y_t) / p(y_{t+1} | y_t) )

    Args:
        x, y : 1D numpy arrays (same length).
        bins : number of bins for discretization.
        lag  : time shift for 'x' relative to 'y_{t+1}'.
        eps  : small constant to avoid log(0).

    Returns:
        TE in bits (since we use log base 2).
    """
    length = len(x)
    if len(y) != length:
        raise ValueError("x and y must have the same length.")
    if length <= lag:
        raise ValueError("Time series too short for the given lag.")

    # We'll consider points from t = 0..(length - lag - 1)
    # so that y_{t+1} is valid, x_t at t, etc.
    if length - lag - 1 <= 0:
        return np.nan  # not enough data

    x_t = x[:-lag-1]         
    y_t = y[:-lag-1]
    y_next = y[1:-lag]       

    # Discretize each
    x_bin = discretize(x_t, bins)
    y_bin = discretize(y_t, bins)
    yn_bin = discretize(y_next, bins)

    # Combine into a single array for 3D histogram
    data_3d = np.vstack([x_bin, y_bin, yn_bin]).T  # shape (num_points, 3)

    counts_3d, edges = np.histogramdd(data_3d, bins=(bins, bins, bins))
    total_count = np.sum(counts_3d)
    if total_count == 0:
        return np.nan

    p_xyz = counts_3d / total_count  # p(x_t, y_t, y_{t+1})

    # We also need:
    #  p(x_t, y_t) = sum over y_{t+1}
    #  p(y_t, y_{t+1}) = sum over x_t
    p_xy = np.sum(p_xyz, axis=2)  # shape (bins, bins)
    p_yy = np.sum(p_xyz, axis=0)  # shape (bins, bins)
    p_y  = np.sum(p_yy, axis=1)   # shape (bins,)

    te_val = 0.0
    for i_x in range(bins):
        for i_y in range(bins):
            for i_yn in range(bins):
                p_val = p_xyz[i_x, i_y, i_yn]
                if p_val <= 0:
                    continue

                p_xy_val = p_xy[i_x, i_y]
                if p_xy_val <= 0:
                    continue

                p_yy_val = p_yy[i_y, i_yn]
                if p_y[i_y] <= 0:
                    continue

                # p(y_{t+1} | x_t, y_t)
                p_cond1 = p_val / p_xy_val
                # p(y_{t+1} | y_t)
                p_cond2 = p_yy_val / p_y[i_y]

                te_val += p_val * np.log2((p_cond1 + eps) / (p_cond2 + eps))

    return te_val

def measure_spatial_te(field, bins=16, lag=1):
    """
    Compute pairwise Transfer Entropy between all pairs of spatial locations 
    in a 2D field (across time).

    Args:
        field: 3D np.array, shape = (T, Nx, Ny). 
               field[t, i, j] = value at time t, spatial coords (i, j).
        bins : number of bins for discretization in TE.
        lag  : temporal lag for TE (X_t -> Y_{t+lag}).

    Returns:
        te_matrix: 4D np.array, shape = (Nx, Ny, Nx, Ny), 
                   where te_matrix[i, j, k, l] = TE from (i, j)->(k, l).
    """
    T, Nx, Ny = field.shape
    te_matrix = np.zeros((Nx, Ny, Nx, Ny), dtype=np.float64)

    for i in tqdm(range(Nx), desc="Row i"):
        for j in range(Ny):
            source_series = field[:, i, j]  # shape (T,)

            for k in range(Nx):
                for l in range(Ny):
                    if (i == k) and (j == l):
                        # Optionally skip self-entropy
                        te_matrix[i, j, k, l] = 0.0
                        continue

                    target_series = field[:, k, l]  # shape (T,)

                    # Compute TE: source -> target
                    te_val = transfer_entropy_continuous(
                        source_series, target_series,
                        bins=bins, lag=lag
                    )

                    # If TE is NaN (insufficient data, etc.), set 0
                    te_matrix[i, j, k, l] = 0.0 if np.isnan(te_val) else te_val

    return te_matrix
     
    	
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

print("Loaded press_fields: ",press_fields.shape)
te_mat = measure_spatial_te(press_fields, bins=8, lag=1)
print(te_mat)
print("done with TE")























    

