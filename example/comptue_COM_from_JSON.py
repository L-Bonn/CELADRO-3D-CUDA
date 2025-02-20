import sys

sys.path.insert(0, "celadro_3D_scripts_final/plot/")
import numpy as np
import matplotlib.pyplot as plts
import archive
import gc

##################################################
# Init

if len(sys.argv) == 1:
    print("Please provide an input file.")
    exit(1)

# load archive from file
ar = archive.loadarchive(sys.argv[1])

oname = ""
if len(sys.argv) == 3:
    oname = "movie_"+sys.argv[2]
    print("Output name is", sys.argv[2])
    

def _get_field(phases, vals, size=1, mode='wrap'):
    """
    Compute the coarse grained field from a collection of phase-fields and
    associated values: ret[i] = sum_j phases[j]*values[i, j].

    Args:
        phases: List of phase-fields.
        vals: List of lists of size (None, len(phases)) of values to be
            associated with each phase-field.
        size: Coarse-graining size.
        mode: How to treat the boundaries, see
            scipy.ndimage.filters.uniform_filter.

    Returns:
        A list of fields, each corresponding to the individual values.
    """
    ret = []
    for vlist in vals:
        assert len(vlist) == len(phases)
        field = np.zeros(phases[0].shape)
        for n in range(len(phases)):
            field += vlist[n]*phases[n]
        field = ndimage.filters.uniform_filter(field, size=size, mode=mode)
        ret.append(field)
    return ret
 
def writeVTK(frame,fr):
    #N = (frame.parameters['Size'][0]*frame.parameters['Size'][1]*frame.parameters['Size'][2])
    #p = np.zeros(N)
    #p = np.reshape(p,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
    for i in range(len(frame.phi)): 
    	psum = 0
    	tmpx = 0
    	tmpy = 0
    	tmpz = 0
    	p = frame.phi[i]
    	for z in np.arange(0,frame.parameters['Size'][2],1):
    		for x in np.arange(0,frame.parameters['Size'][0],1):
    			for y in np.arange(0,frame.parameters['Size'][1],1):
    				tmpx += p[z,x,y] * x
    				tmpy += p[z,x,y] * y
    				tmpz += p[z,x,y] * z
    				psum += p[z,x,y]
    	print(fr,' ',i,' ',tmpx/psum,' ',tmpy/psum,' ',tmpz/psum);
    

rng = np.arange(1,ar._nframes+1,1)
rng = np.arange(0,20,1)
for fr in (rng):
    frame = ar.read_frame(fr)
    writeVTK(frame,fr)
    del frame
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
