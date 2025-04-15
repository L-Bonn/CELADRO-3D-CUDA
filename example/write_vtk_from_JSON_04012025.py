#!/usr/bin/env python
import vtk
import sys

sys.path.insert(0, "celadro_3D_scripts_final/plot/")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import plot
import archive
import animation
import matplotlib.cm as cm  # for jet colormap


if len(sys.argv) == 1:
    print("Please provide an input file.")
    exit(1)

# Load archive from file.
ar = archive.loadarchive(sys.argv[1])

oname = ""
if len(sys.argv) == 3:
    oname = "movie_" + sys.argv[2]
    print("Output name is", sys.argv[2])


props = []
# Here we loop over frames 1 to 20; adjust rng as needed.
rng = np.arange(1, 255, 1)
# Or: rng = np.arange(1, ar._nframes + 1)

for fr in rng:
    frame = ar.read_frame(fr)
    props.extend(frame.stored_gam)

# Round to avoid floating point issues and get unique gamma values.
unique_props = np.unique(np.round(props, decimals=6))
prop_to_mapped_val = {}
base = 2
for i, prop in enumerate(unique_props):
    prop_to_mapped_val[prop] = base + i * 3

def writeVTK(fname, frame, prop_val, prop_id):
    # Total number of points.
    N = (frame.parameters['Size'][0] *
         frame.parameters['Size'][1] *
         frame.parameters['Size'][2])
    p = np.zeros(N)
    p = np.reshape(
        p,
        (frame.parameters['Size'][2],
         frame.parameters['Size'][0],
         frame.parameters['Size'][1])
    )
    cId = np.zeros(N)
    cId = np.reshape(
        cId,
        (frame.parameters['Size'][2],
         frame.parameters['Size'][0],
         frame.parameters['Size'][1])
    )
    
    # Use the global mapping: note we round each gamma value to 6 decimals.
    for i in range(len(frame.phi)):
        if (frame.stored_gam[i] == prop_val):# and ((frame.com[i][2] - 8) < 2.0 * 8):
            p += frame.phi[i]
            cphi = frame.phi[i]
            #cphi = np.where(cphi < 0.5, 0, cphi)
            #cphi = prop_id * cphi
            cId += cphi

    # If needed, get velocity (defined elsewhere).
    bc_mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    
    with open(fname, "w+") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("volume example\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        # VTK expects dimensions as: DIMENSIONS (x, y, z)
        f.write("DIMENSIONS {} {} {}\n".format(
            frame.parameters['Size'][1],
            frame.parameters['Size'][0],
            frame.parameters['Size'][2])
        )
        f.write("ASPECT_RATIO 1 1 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("POINT_DATA {}\n".format(N))
        f.write("SCALARS volume_scalars double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for z in range(frame.parameters['Size'][2]):
            for x in range(frame.parameters['Size'][0]):
                for y in range(frame.parameters['Size'][1]):
                    f.write("%.6f\n" % cId[z, x, y])
        # Additional fields (e.g., velocities) could go here if desired.
        # f.write(...)
        f.close()

for fr in rng:
    frame = ar.read_frame(fr)
    lprops = frame.stored_gam
    ulprops = np.unique(lprops)
    for j in range(len(ulprops)):
        fin = "frame_" + str(fr) + "_type_" + str(j) + ".vtk"
        print("time step:", fr," ulprop: ",ulprops, flush=True)
        writeVTK(fin, frame, ulprops[j], prop_to_mapped_val[ulprops[j]])

