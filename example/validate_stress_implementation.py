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
print('running write_force_fields_12012023.py',flush=True)

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit(1)
    
ar = archive.loadarchive(sys.argv[1])

def coarse_grain_stress_field_3d_07052022(cgLz,cgLy,cgLx,LZ,LY,LX,fx,fy,fz):

	nvx = LX-1;
	nvy = LY-1;
	nvz = LZ-1;
			
	nsiteX = int(2)
	nsiteY = int(2)
	nsiteZ = int(2)

	sigXX = np.zeros(((nvz),(nvx),(nvy)))
	sigXY = np.zeros(((nvz),(nvx),(nvy)))
	sigXZ = np.zeros(((nvz),(nvx),(nvy)))
	sigYX = np.zeros(((nvz),(nvx),(nvy)))
	sigYY = np.zeros(((nvz),(nvx),(nvy)))
	sigYZ = np.zeros(((nvz),(nvx),(nvy)))
	sigZX = np.zeros(((nvz),(nvx),(nvy)))
	sigZY = np.zeros(((nvz),(nvx),(nvy)))
	sigZZ = np.zeros(((nvz),(nvx),(nvy)))
	
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	intX = np.zeros(2)
	intY = np.zeros(2)
	intZ = np.zeros(2)
	
	for i in range(nvx):
		LengthX[i] = i * (1)
	for i in range(nvy):
		LengthY[i] = i * (1)
	for i in range(nvz):
		LengthZ[i] = i * (1)
	
	for z in range(nvz):
		lbz = LengthZ[z]
		for x in range(nvx):
			lbx = LengthX[x]
			for y in range(nvy):
				lby = LengthY[y]
				
				for i in range(nsiteX):
					intX[i] = lbx + i
				for i in range(nsiteY):
					intY[i] = lby + i
				for i in range(nsiteZ):
					intZ[i] = lbz + i

				cc = np.array([np.mean(intX), np.mean(intY), np.mean(intZ)]) 
				sxx = 0
				sxy = 0
				sxz = 0
				syx = 0
				syy = 0
				syz = 0
				szx = 0
				szy = 0
				szz = 0

				for zz in range(nsiteZ):
					for xx in range(nsiteX):
					    for yy in range(nsiteY):

					    	ci = np.array([intX[xx],intY[yy],intZ[zz]])
					    	xci = cc - ci
					    	xci = xci / np.linalg.norm(xci)
					    	
					    	sxx += xci[0] * fx[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	sxy += xci[0] * fy[int(intZ[zz]),int(intX[xx]),int(intY[yy])] + xci[1] * fx[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	sxz += xci[0] * fz[int(intZ[zz]),int(intX[xx]),int(intY[yy])] + xci[2] * fx[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	syx += xci[1] * fx[int(intZ[zz]),int(intX[xx]),int(intY[yy])] + xci[0] * fy[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	syy += xci[1] * fy[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	syz += xci[1] * fz[int(intZ[zz]),int(intX[xx]),int(intY[yy])] + xci[2] * fy[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	szx += xci[2] * fx[int(intZ[zz]),int(intX[xx]),int(intY[yy])] + xci[0] * fz[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	szy += xci[2] * fy[int(intZ[zz]),int(intX[xx]),int(intY[yy])] + xci[1] * fz[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
					    	szz += xci[2] * fz[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
				
				sxx = sxx/(nsiteX*nsiteY*nsiteZ)
				sxy = sxy/(2.*nsiteX*nsiteY*nsiteZ)
				sxz = sxz/(2.*nsiteX*nsiteY*nsiteZ)
				syx = syx/(2.*nsiteX*nsiteY*nsiteZ)
				syy = syy/(nsiteX*nsiteY*nsiteZ)
				syz = syz/(2.*nsiteX*nsiteY*nsiteZ)
				szx = szx/(2.*nsiteX*nsiteY*nsiteZ)
				szy = szy/(2.*nsiteX*nsiteY*nsiteZ)
				szz = szz/(nsiteX*nsiteY*nsiteZ)
                
				sigXX[z,x,y] = sxx    
				sigXY[z,x,y] = sxy  
				sigXZ[z,x,y] = sxz   
				sigYY[z,x,y] = syy  
				sigYZ[z,x,y] = syz  
				sigZZ[z,x,y] = szz  

	xsp = np.linspace(0, LX-1, (nvx-1))
	ysp = np.linspace(0, LY-1, (nvy-1))
	zsp = np.linspace(0, LZ-1, (nvz-1))
    	
	return sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ
    
def coarse_grain_stress_field(fx,fy,fz,cgLz,cgLy,cgLx):

    (LZ, LX, LY) = fx.shape
    sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ = coarse_grain_stress_field_3d_07052022(cgLz,cgLy,cgLx,LZ,LY,LX,fx,fy,fz)
    return sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ#,xsp,ysp,zsp
   
    
def ipx(x,y,z,Lx,Ly): 
	return (y + Ly * x + Lx*Ly*z);
	
def convert_to_field(data,nx,ny,nz,xsec):
	field = np.zeros((nx,ny))
	for k in range(nz):
		for i in range(nx):
			for j in range(ny):
				idx = ipx(i,j,k,nx,ny)
				if(k == xsec):
					field[i,j] = data[idx]
	return field
	
	
	
xsec = 8
size = 1
tstep = 10

frame = ar.read_frame(int(tstep))
mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
fx, fy, fz = plot.get_force_field(frame.phi, frame.parameters['xi']*frame.velocity, size, mode=mode)
fx *= (1.-frame.parameters['walls'])
fy *= (1.-frame.parameters['walls'])
fz *= (1.-frame.parameters['walls'])

sxx,syy,szz,sxy,sxz,syz = coarse_grain_stress_field(fx,fy,fz,1,1,1) 
p = (1/3)*(sxx+syy+szz);

sxx_new = frame.field_sxx
sxx_new = np.reshape(sxx_new,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
syy_new = frame.field_syy
syy_new = np.reshape(syy_new,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
szz_new = frame.field_szz
szz_new = np.reshape(szz_new,(frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
p_new = (1/3)*(sxx_new+syy_new+szz_new)
print("old: ",sxx.shape," new: ",sxx_new.shape)

cSxx = frame.cSxx
print(cSxx)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # adjust figure size as needed

# Plot first panel (new)
ax = axs[0]
im = ax.imshow(sxx_new[xsec, :, :], interpolation='lanczos', cmap='plasma', origin='lower')
ax.set_title('(new)')
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.3)
fig.colorbar(im, cax=cax, orientation='horizontal')

# Plot second panel (old)
ax = axs[1]
im = ax.imshow(sxx[xsec, :, :], interpolation='lanczos', cmap='plasma', origin='lower')
ax.set_title('(old)')
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.3)
fig.colorbar(im, cax=cax, orientation='horizontal')

plt.tight_layout()
plt.show()
























    

