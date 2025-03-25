#!/usr/bin/env python3

import sys
sys.path.insert(0, "celadro_3D_scripts_final/plot/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # for standard colormaps
import vtk

from scipy import ndimage
import plot
import archive
import animation



# If you have a custom 'archive' module, import it here.
# E.g.:
import archive

# Check usage
if len(sys.argv) == 1:
    print("Usage: {} <archive_file>".format(sys.argv[0]))
    sys.exit(1)

archive_file = sys.argv[1]
ar = archive.loadarchive(archive_file)

##############################################################################
# Step A: Gather all properties from all frames.
##############################################################################
all_props = []
rng = np.arange(1, 258, 1)  # for demonstration; or use ar._nframes

for fr in rng:
    frame = ar.read_frame(fr)
    all_props.extend(frame.stored_gam)

# Round and extract unique property values
all_props = np.unique(np.round(all_props, decimals=6))
print("Found {} unique gamma values overall.".format(len(all_props)))

# Build a colormap for all unique props
cmap = cm.get_cmap('jet', len(all_props))
color_map = {}
for i, prp in enumerate(all_props):
    t = i / max(1, (len(all_props) - 1))  # normalized 0..1
    r, g, b, _ = cmap(t)
    color_map[prp] = (r, g, b)

prop_to_mapped_val = {}
base = 2
for i, prop in enumerate(all_props):
    prop_to_mapped_val[prop] = base + i * 3
##############################################################################
# Step B: Define a volume rendering function that can handle multiple files
#         in a single scene, each with a different property/color if desired.
##############################################################################
def VolRendering(vtk_files, out_png, props, color_map,id_list):
    """
    Render multiple VTK files (one per property) in the same scene and write to PNG.
    
    Arguments:
      vtk_files -- list of .vtk filenames
      out_png   -- output PNG filename
      props     -- list of property values associated with each file (same length as vtk_files)
      color_map -- dict mapping property -> (r, g, b)
    """
    # Create a renderer
    ren1 = vtk.vtkRenderer()
    ren1.SetBackground(1.0, 1.0, 1.0)  # White background

    # For each file, create a volume actor
    for i, fin in enumerate(vtk_files):
        prop_val = props[i]
        # Look up the color for this property
        if prop_val in color_map:
            r, g, b = color_map[prop_val]
        else:
            r, g, b = (0.8, 0.8, 0.8)  # fallback color

        # Read data
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(fin)
        reader.Update()

        # Build opacity function
        opacityTF = vtk.vtkPiecewiseFunction()
        # Example: data range in [0.5, 1.0] -> [0, 1.1] opacity
        # Adjust as needed for your data
        opacityTF.AddPoint(0.5, 0.0)
        opacityTF.AddPoint(1.0, 1.1)

        # Build color transfer function
        colorTF = vtk.vtkColorTransferFunction()
        
        # Optionally, map entire data range to one color
        data_range = reader.GetOutput().GetScalarRange()
        colorTF.AddRGBPoint(0., r, g, b)
        #colorTF.AddRGBPoint(data_range[0], r, g, b)
        #colorTF.AddRGBPoint(data_range[1], r, g, b)
        #colorTF.AddRGBPoint(0., 255/255, 153/255, 51/255);#wt - orange 
        # Set volume properties
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTF)
        volumeProperty.SetScalarOpacity(opacityTF)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        # GPU Ray Cast Mapper
        volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        volumeMapper.SetInputConnection(reader.GetOutputPort())

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        # Add the volume to the renderer
        ren1.AddVolume(volume)

    # Set up camera
    camera = vtk.vtkCamera()
    ren1.SetActiveCamera(camera)
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(64, 300, 200)
    camera.SetFocalPoint(64, 64, 0)
    camera.Zoom(1)
    camera.SetClippingRange(0.1, 12000)

    # Off-screen render
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.SetOffScreenRendering(1)
    renWin.SetSize(750, 750)
    renWin.Render()

    # Capture image
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renWin)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(out_png)
    writer.SetInputData(w2i.GetOutput())
    writer.Write()

##############################################################################
# Step C: For each frame, figure out which properties appear, produce
#         one .vtk file for each, and render them all together in one PNG.
##############################################################################
for fr in rng:
    frame = ar.read_frame(fr)
    lprops = np.round(frame.stored_gam, decimals=6)
    ulprops = np.unique(lprops)

    # Build a list of .vtk files for these properties
    fins = []
    prop_list = []
    id_list = []
    for j, prop_val in enumerate(ulprops):
        fin = f"frame_{fr}_type_{j}.vtk"
        fins.append(fin)
        prop_list.append(prop_val)
        id_list.append(prop_to_mapped_val[prop_val])

    out_png = f"config_{fr}.png"
    print(f"Rendering frame {fr} -> {out_png}")

    # Render all volumes in one pass
    VolRendering(fins, out_png, prop_list, color_map,id_list)

