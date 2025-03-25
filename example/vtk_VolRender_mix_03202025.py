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

###############################################################################
# Step 1: Build global gamma → mapped_val mapping across all frames
###############################################################################
all_gammas = []

# Here we loop over frames 1 to 9; adjust rng as needed.
rng = np.arange(1, 122, 1)
#rng = np.arange(1, ar._nframes + 1)
for fr in rng:
    frame = ar.read_frame(fr)
    all_gammas.extend(frame.stored_gam)

# Round to avoid floating point issues and get unique gamma values.
all_gammas = np.unique(np.round(all_gammas, decimals=6))

gamma_to_mapped_val = {}
base = 2
for i, gamma in enumerate(all_gammas):
    gamma_to_mapped_val[gamma] = base + i * 15

###############################################################################
# Step 2: Build global color transfer function based on the mapping using jet colormap
###############################################################################
colorTransferFunction = vtk.vtkColorTransferFunction()
n_colors = len(gamma_to_mapped_val)
for i, gamma in enumerate(all_gammas):
    mapped_val = gamma_to_mapped_val[gamma]
    # Normalize index between 0 and 1.
    t = i / float(n_colors - 1) if n_colors > 1 else 0.5
    #r, g, b, _ = cm.jet(t)
    cmap = cm.get_cmap('prism')
    r, g, b, _ = cmap(t)
    colorTransferFunction.AddRGBPoint(mapped_val, r, g, b)

###############################################################################
# Step 3: Define the volume rendering function that uses the global color table
###############################################################################
def VolRendering(fin, fout, gam_vec):
    colors = vtk.vtkNamedColors()
    ren1 = vtk.vtkRenderer()

    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(fin)

    # Create opacity transfer function: span the range of mapped gamma values.
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    min_val = min(gamma_to_mapped_val.values())
    max_val = max(gamma_to_mapped_val.values())
    opacityTransferFunction.AddPoint(min_val, 0.0)
    opacityTransferFunction.AddPoint(max_val, 1.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('White'))

    camera = vtk.vtkCamera()
    ren1.SetActiveCamera(camera)
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(64, 300, 200)
    camera.SetFocalPoint(64, 64, 0)
    camera.Zoom(1)
    camera.SetClippingRange(0.1, 12000)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(ren1)
    renderWindow.SetOffScreenRendering(1)
    renderWindow.SetSize(750, 750)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fout)
    writer.SetInputData(windowToImageFilter.GetOutput())
    writer.Write()

###############################################################################
# Step 4: Loop over frames and render each using the global mapping
###############################################################################
for fr in rng:
    frame = ar.read_frame(fr)
    print("time step:", fr, flush=True)
    fin = "frame_" + str(fr) + ".vtk"
    fout = "config_" + str(fr) + ".png"
    # Ensure gamma values match the precision used in the global mapping.
    gam_vec = np.round(frame.stored_gam, decimals=6)
    VolRendering(fin, fout, gam_vec)

