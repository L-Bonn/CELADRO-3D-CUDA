#!/usr/bin/env python2
import sys

sys.path.insert(0, "celadro_3D_scripts_final/plot/")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import plot
import archive
import animation

##################################################
# Init

if len(sys.argv) == 1:
    print("Please provide an input file.")
    exit(1)

# load archive from file
ar = archive.loadarchive(sys.argv[1])

oname = ""
if len(sys.argv) == 3:
    oname = "movie_" + sys.argv[2]
    print("Output name is", sys.argv[2])

###############################################################################
# Global mapping: compute the mapping for gamma values over all frames.
###############################################################################
all_gammas = []
# Loop over all frames (using the archive's _nframes)
rng = np.arange(1, ar._nframes + 1)
rng = np.arange(1, 101)
for fr in rng:
    frame = ar.read_frame(fr)
    # Extend our list with this frame's gamma values
    all_gammas.extend(frame.stored_gam)
# Round to 6 decimals to avoid floating point issues, then get unique values.
all_gammas = np.unique(np.round(all_gammas, decimals=6))
global_mapping = {}
base = 2  # starting integer value
for i, gamma in enumerate(all_gammas):
    global_mapping[gamma] = base + i * 15

###############################################################################
# Utility functions.
###############################################################################
def _get_field(phases, vals, size=1, mode='wrap'):
    """
    Compute the coarse grained field from a collection of phase-fields and
    associated values.
    """
    ret = []
    for vlist in vals:
        assert len(vlist) == len(phases)
        field = np.zeros(phases[0].shape)
        for n in range(len(phases)):
            field += vlist[n] * phases[n]
        field = ndimage.filters.uniform_filter(field, size=size, mode=mode)
        ret.append(field)
    return ret

def get_velocity_field(phases, vel, size=1, mode='wrap'):
    """
    Compute coarse-grained velocity field.
    """
    v0 = [v[0] for v in vel]
    v1 = [v[1] for v in vel]
    v2 = [v[2] for v in vel]
    return _get_field(phases, [v0, v1, v2], size, mode)

###############################################################################
# Write the VTK file using the global gamma mapping.
###############################################################################
def writeVTK(fname, frame, cGam):
    # Total number of points.
    N = frame.parameters['Size'][0] * frame.parameters['Size'][1] * frame.parameters['Size'][2]
    p = np.zeros(N)
    p = np.reshape(p, (frame.parameters['Size'][2], frame.parameters['Size'][0], frame.parameters['Size'][1]))
    cId = np.zeros(N)
    cId = np.reshape(cId, (frame.parameters['Size'][2], frame.parameters['Size'][0], frame.parameters['Size'][1]))
    
    # Use the global mapping: note we round each gamma value to 6 decimals.
    for i in range(len(frame.phi)):
        p += frame.phi[i]
        cphi = frame.phi[i]
        # Only consider regions where the phase field is active (>= 0.5).
        cphi = np.where(cphi < 0.5, 0, cphi)
        mapped_val = global_mapping[round(cGam[i], 6)]
        cphi = mapped_val * cphi
        if (frame.com[i][2]-8) < 2. * 8:
        	cId += cphi

    bc_mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    vx, vy, vz = get_velocity_field(frame.phi, frame.velocity, 1, mode=bc_mode)
    
    with open(fname, "w+") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("volume example\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        # VTK expects dimensions as: DIMENSIONS (x, y, z)
        f.write("DIMENSIONS {} {} {}\n".format(frame.parameters['Size'][1],
                                                 frame.parameters['Size'][0],
                                                 frame.parameters['Size'][2]))
        f.write("ASPECT_RATIO 1 1 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("POINT_DATA {}\n".format(N))
        f.write("SCALARS volume_scalars double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for z in np.arange(0, frame.parameters['Size'][2], 1):
            for x in np.arange(0, frame.parameters['Size'][0], 1):
                for y in np.arange(0, frame.parameters['Size'][1], 1):
                    f.write("%.6f\n" % cId[z, x, y])
        f.close()

###############################################################################
# Main loop over frames.
###############################################################################
rng = np.arange(1, ar._nframes + 1)
#rng = np.arange(1, 101)
for fr in rng:
    fname = 'frame_' + str(fr) + '.vtk'
    print(fname, flush=True)
    frame = ar.read_frame(fr)
    gam_vec = frame.stored_gam  # use the gamma vector from the frame
    print(gam_vec)
    writeVTK(fname, frame, gam_vec)

