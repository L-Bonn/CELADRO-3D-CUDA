import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import ndimage
from itertools import product
from math import log
from numpy import linalg as LA
import math 
import sys
import matplotlib.cm
sys.path.insert(0, "celadro_3D_scripts_final/plot/")
import plot
import archive
import animation
print('running write_force_fields_12012023.py',flush=True)

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit(1)

def plot_cell_contours(ax,frame,z_section,levels,main_cmap='brg'):
	for i in range(len(frame.phi)):   
		p_main = frame.phi[i][z_section,:,:]
		ax.contour(np.arange(frame.parameters['Size'][1]),
                         np.arange(frame.parameters['Size'][0]),
                         p_main, levels=levels, cmap=main_cmap)
   
def plot_cell_stress_dir(ax,frame):
	for i in range(len(frame.phi)):
		sxx = frame.cSxx[i]
		sxy = frame.cSxy[i]
		syy = frame.cSyy[i]
		stress = np.array([[sxx,sxy],[sxy,syy]])
		eigvals, eigvecs = LA.eig(stress)
		idx = eigvals.argsort()[::-1]   
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:,idx]
		maxIndex = np.where(eigvals == np.amax(eigvals))
		maxEigVec = eigvecs[:,maxIndex].ravel()
		norm = np.linalg.norm(maxEigVec)
		nx = maxEigVec[1]
		ny = maxEigVec[0]
		c = frame.com[i]
		a = norm
		ax.arrow(c[1], c[0],  a*ny,  a*nx)
		ax.arrow(c[1], c[0], -a*ny, -a*nx)    
        
        
xsec = 8
ar = archive.loadarchive(sys.argv[1])
frames = np.arange(1,ar._nframes+1,1)
for fr in frames: 
	frame = ar.read_frame(fr)
	fig, ax = plt.subplots(figsize=(8, 8))
	plot_cell_contours(ax,frame,xsec,[0.5])
	plot_cell_stress_dir(ax,frame)
	
	ax.set_aspect('equal', adjustable='box')
	ax.set_xlim([0, frame.parameters['Size'][0] - 1])
	ax.set_ylim([0, frame.parameters['Size'][1] - 1])
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.axis('off')  # show axes
	fig.tight_layout()
	fname = 'maps_' + str(fr) + '.png'
	plt.savefig(fname, dpi=300, bbox_inches="tight")
	print("done with ",fname)
	#plt.close()
	del frame
	#plt.show()





















    

