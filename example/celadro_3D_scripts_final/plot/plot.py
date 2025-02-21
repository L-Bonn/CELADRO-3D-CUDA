import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, atan2, cos, sin
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from itertools import product
from matplotlib.streamplot import streamplot
from numpy import linalg as LA
import matplotlib as mpl
from matplotlib import cm
from scipy.stats import kurtosis, skew
from matplotlib import colors
import matplotlib
import math

def round_sig(x, sig=2):
	return round(x, sig-int(floor(log10(abs(x))))-1)
	
def _update_dict(d, k, v):
    """
    Update dictionary with k:v pair if k is not in d.

    Args:
        d: The dictionary to update.
        k, v: The key/value pair.
    """
    if k not in d:
        d[k] = v
              
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


def get_velocity_field(phases, vel, size=1, mode='wrap'):
    """
    Compute coarse-grained nematic field.

    Args:
        phases: List of phase fields.
        vel: List of 2d velocities associated with each phase field.
        size: Coarse-graining size.
        mode: How to treat the boundaries, see
            scipy.ndimage.filters.uniform_filter.
    """
    v0 = [v[0] for v in vel]
    v1 = [v[1] for v in vel]
    v2 = [v[2] for v in vel]
    return _get_field(phases, [v0, v1, v2], size, mode)
    
    
def get_force_field(phases, fvec, size=1, mode='wrap'):
    """
    Compute coarse-grained nematic field.

    Args:
        phases: List of phase fields.
        vel: List of 2d velocities associated with each phase field.
        size: Coarse-graining size.
        mode: How to treat the boundaries, see
            scipy.ndimage.filters.uniform_filter.
    """
    f0 = [f[0] for f in fvec]
    f1 = [f[1] for f in fvec]
    f2 = [f[2] for f in fvec]
    return _get_field(phases, [f0, f1, f2], size, mode)   
    
    
def get_stress_field(sig, size=1, mode='wrap'):
    """
    Compute coarse-grained stress field.

    Args:
        phases: List of phase fields.
        vel: List of 2d velocities associated with each phase field.
        size: Coarse-graining size.
        mode: How to treat the boundaries, see
            scipy.ndimage.filters.uniform_filter.
    """

    field = ndimage.filters.uniform_filter(sig, size=size, mode=mode)
    return field 
    
    
    
def get_nematic_field(phases, qxx, qxy, qxz, qyy, qyz, qzz, size=1, mode='wrap'):
    """
    Compute coarse-grained nematic field.

    Args:
        phases: List of phase fields.
        qxx, qxy: Components of the nematic field of the individual phase
            fields.
        size: Coarse-graining size.
        mode: How to treat the boundaries, see
            scipy.ndimage.filters.uniform_filter.
    """

    return _get_field(phases, [qxx, qxy, qxz, qyy, qyz, qzz], size, mode)
    
    
def get_vorticity_field(phases, vort, size=1, mode='wrap'):
    """
    Compute coarse-grained nematic field.

    Args:
        phases: List of phase fields.
        vel: List of 2d velocities associated with each phase field.
        size: Coarse-graining size.
        mode: How to treat the boundaries, see
            scipy.ndimage.filters.uniform_filter.
    """
    v0 = [v[0] for v in vort]
    v1 = [v[1] for v in vort]
    v2 = [v[2] for v in vort]
    return _get_field(phases, [v0, v1, v2], size, mode) 
    
        


def cell(frame, i, zxsec, engine=plt, **kwargs):
    """
    Plot a single phase field as a contour.

    Args:
        frame: Frame to plot, from archive module.
        i: Index of the cell to plot.
        engine: Plotting engine or axis.
        color: Color to use for the contour.
        **kwargs: Keyword arguments passed to contour().
    """
    p = frame.phi[i]
    p = p[zxsec,:,:]

    _update_dict(kwargs, 'color', 'k')
    _update_dict(kwargs, 'levels', [.5])

    engine.contour(np.arange(0, frame.parameters['Size'][1]),
                   np.arange(0, frame.parameters['Size'][0]),
                   p, linewidths =[0.5],**kwargs)


def cells(frame, zxsec, engine=plt, colors='k'):
    """
    Plot all cells as contours.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
        colors: Colors to use for the contour. Can also be a list of colors,
            one for each cell.
    """
    if not isinstance(colors, list):
        colors = len(frame.phi)*[colors]

    for i in range(len(frame.phi)):
        cell(frame, i, zxsec, engine, color='k')

def cells_attached(frame, zxsec,cells_vec,engine=plt, colors='k'):
    """
    Plot all cells as contours.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
        colors: Colors to use for the contour. Can also be a list of colors,
            one for each cell.
    """
    if not isinstance(colors, list):
        colors = len(frame.phi)*[colors]

    #for i in range(len(frame.phi)):
    for i in cells_vec:
        cell(frame, i, zxsec, engine, color='k')
        
        
def phase(frame, zxsec, engine=plt ):
    """
    Plot single phase as a density plot.

    Args:
        frame: Frame to plot, from archive module.
        n: Index of the cell to plot.
        engine: Plotting engine or axis.
        cbar: Display cbar?
    """
    p = np.zeros((frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
    for i in range(len(frame.phi)):
    	p += frame.phi[i]

    engine.imshow(p[zxsec,:,:], interpolation='lanczos', cmap='Greys', origin='lower')       
    
    
def cell_contour(frame, i, zxsec, engine=plt):
    """
    Plot a single phase field as a contour.

    Args:
        frame: Frame to plot, from archive module.
        i: Index of the cell to plot.
        engine: Plotting engine or axis.
        color: Color to use for the contour.
        **kwargs: Keyword arguments passed to contour().
    """
    cmap = mpl.cm.get_cmap('jet')
    cLevel = 4
    rng = np.arange(0,cLevel,1)
    
    p = frame.phi[i]
    for j in (rng):
        zsec = zxsec + j
        pc = p[zsec,:,:]

        cax = engine.contour(np.arange(0, frame.parameters['Size'][1]),
                   np.arange(0, frame.parameters['Size'][0]),
                   pc, levels=[0.5],colors=[cmap(j/(cLevel-1))],linewidths =[0.5],vmin=10,vmax=13 )
           
                            

def cells_contour(frame, zxsec, engine=plt):

    for i in range(len(frame.phi)):
        cell_contour(frame, i, zxsec,engine)
        
        
        
        
        
        
def substrate(frame, zxsec, engine=plt):
    
    wall = frame.parameters['walls']
    wall = wall[zxsec,:,:] 
    cax = engine.imshow(wall, interpolation='lanczos', cmap='viridis', origin='lower')
    plt.colorbar(cax)


        
def interfaces(frame, zxsec, engine=plt):
    """
    Plot the overlap between cells as heatmap using beautiful shades of gray
    for an absolutely photorealistic effect that will impress all your friends.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
    """
    totphi = np.zeros((frame.parameters['Size'][2],frame.parameters['Size'][0],frame.parameters['Size'][1]))
    for i in range(len(frame.phi)):
        for j in range(i+1, len(frame.phi)):
            totphi += frame.phi[i]*frame.phi[j]

    cmap = LinearSegmentedColormap.from_list('mycmap', ['grey', 'white'])
    engine.imshow(totphi[zxsec,:,:], interpolation='lanczos', cmap=cmap, origin='lower')      
    
      
      
      
      
        
    

def fill_cell(frame,fr,lprop,i,zxsec,engine=plt):

    p = frame.phi[i]
    p = p[zxsec,:,:]
    
    unqprops = np.unique(lprop)
    number_of_bins = len(unqprops)


    color = ['#edeef5','#bcbcbc','#000000','#6f03fc','#FFC300','#FF5733','#bf0823','#581845']
    #cmap = matplotlib.cm.get_cmap('rainbow')
    #color = []
    #for j in np.arange(0,number_of_bins+1,1):
    #	color.append(	matplotlib.colors.to_hex(	cmap(j/number_of_bins)	))
    
    ncells = len(frame.phi)
    tprop = lprop[fr*ncells:(fr+1)*ncells:1];
    map_value = np.where(tprop[i]==unqprops)[0]
    map_value = map_value[0]
    map_color = color[int(map_value)]

    engine.contourf(np.arange(0, frame.parameters['Size'][1]),
                   np.arange(0, frame.parameters['Size'][0]),
                   p, linewidths =[0.5],colors=map_color,levels=[0.5,1.5],alpha=0.8)
                   
                           
def fill_cell_prop(frame,fr,lprop,zxsec, engine=plt ):
	for i in range(len(frame.phi)):
		fill_cell(frame,fr,lprop,i,zxsec, engine) 


def outline_cell(frame, lprop, bnds,i,zxsec,engine=plt):

    p = frame.phi[i]
    phi_cell = p
    p = p[zxsec,:,:]
    phi_cell[phi_cell < 0.5 ] = 0
    field = frame.com[i][2]*phi_cell
    hs = field[np.nonzero(field)]
    hmin = np.min(hs)
    
    #cmap = matplotlib.cm.get_cmap('rainbow')
    #color = []
    #for j in np.arange(0,bnds+1,1):
    #	color.append(	matplotlib.colors.to_hex(	cmap(j/bnds)	))

    #map_color = color[int(lprop[i])]
    colors = ['r','r','b']
    map_color = colors[int(lprop[i])]
    if(hmin < frame.parameters['wall_thickness']):
    	engine.contour(np.arange(0, frame.parameters['Size'][1]),
                   np.arange(0, frame.parameters['Size'][0]),
                   p, linewidths =[0.3],colors=map_color,alpha=1,levels=[.5])


def outline_cell_prop(frame,lprop,bnds,zxsec, engine=plt ):
	for i in range(len(frame.phi)):
		outline_cell(frame,lprop,bnds,i,zxsec, engine) 
		
		
		
		
		
        
        
def velocity_field(frame, zxsec, size, avg, minVal,maxVal,engine=plt, magn=True, quiv=False, cbar=False):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        engine: Plotting engine or axis.
        magn: Plot velocity magnitude as a heatmap?
        cbar: Show color bar?
        avg: Size of the averaging (drops points)
    """
    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    vx, vy, vz = get_velocity_field(frame.phi, frame.velocity, size, mode=mode)
    vx *= (1.-frame.parameters['walls'])
    vy *= (1.-frame.parameters['walls'])
    vz *= (1.-frame.parameters['walls'])

    if magn:
        m = np.sqrt(vx**2 + vy**2 + vz**2)
        print('min = ',np.min(vz[zxsec,:,:]))
        print('max = ',np.max(vz[zxsec,:,:]))

        cax = engine.imshow(vz[zxsec,:,:], interpolation='lanczos', cmap='RdBu', origin='lower',vmin=minVal,vmax=maxVal)
        if cbar:
            plt.colorbar(cax)


    if quiv:
    	vx = vx[zxsec,:,:];
    	vy = vy[zxsec,:,:];

    	vx = vx.reshape((vx.shape[0]//avg, avg, vx.shape[1]//avg, avg))
    	vx = np.mean(vx, axis=(1, 3))
    	vy = vy.reshape((vy.shape[0]//avg, avg, vy.shape[1]//avg, avg))
    	vy = np.mean(vy, axis=(1, 3))

    	cax = engine.quiver(np.arange(0, frame.parameters['Size'][1], step=avg),
                        np.arange(0, frame.parameters['Size'][0], step=avg),
                        vy, vx,
                        pivot='tail', units='dots', scale_units='dots', color='r')     
        
        
def velocity_field_streamlines(frame, zxsec, size, avg, engine=plt, magn=True, cbar=False):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        engine: Plotting engine or axis.
        magn: Plot velocity magnitude as a heatmap?
        cbar: Show color bar?
        avg: Size of the averaging (drops points)
    """
    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    vx, vy, vz = get_velocity_field(frame.phi, frame.velocity, size, mode=mode)
    vx *= (1.-frame.parameters['walls'])
    vy *= (1.-frame.parameters['walls'])
    vz *= (1.-frame.parameters['walls'])
    
    m = np.sqrt(vx**2 + vy**2 + vz**2)
    vx = vx[zxsec,:,:];
    vy = vy[zxsec,:,:];
    m = m[zxsec,:,:];
    vx = vx.reshape((vx.shape[0]//avg, avg, vx.shape[1]//avg, avg))
    vx = np.mean(vx, axis=(1, 3))
    vy = vy.reshape((vy.shape[0]//avg, avg, vy.shape[1]//avg, avg))
    vy = np.mean(vy, axis=(1, 3))
    m = m.reshape((m.shape[0]//avg, avg, m.shape[1]//avg, avg))
    m = np.mean(m, axis=(1, 3))                 
                        
    strm = engine.streamplot(np.arange(0, frame.parameters['Size'][1], step=avg),
                        np.arange(0, frame.parameters['Size'][0], step=avg),
                        vy, vx,norm=Normalize,
                        density=1., color=m, arrowsize=0.5, linewidth=0.8, cmap='plasma')
    if cbar:
        plt.colorbar(strm.lines)		   
        
        
def force_field(frame, zxsec, size, engine=plt, magn=True, cbar=False):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        engine: Plotting engine or axis.
        magn: Plot velocity magnitude as a heatmap?
        cbar: Show color bar?
        avg: Size of the averaging (drops points)
    """
    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.Fpressure, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls'])

    if magn:
        m = np.sqrt(fx**2 + fy**2 + fz**2)
        cax = engine.imshow(fz[zxsec,:,:], interpolation='lanczos', cmap='plasma',
                            origin='lower')
        if cbar:
            plt.colorbar(cax)
            
            
def pdf_force_field(frame, size, dims, engine=plt, num_bins =100):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        engine: Plotting engine or axis.
        magn: Plot velocity magnitude as a heatmap?
        cbar: Show color bar?
        avg: Size of the averaging (drops points)
    """
    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.parameters['xi']*frame.velocity-frame.Fpol, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls'])

    (LZ, LX, LY) = fx.shape
    fxs = []
    fys = []
    fzs = []
    for ii in range(len(frame.phi)):
        phi_i = frame.phi[ii]
        for k in range(LZ):
        	for i in range(LX):
        		for j in range(LY):
        			pi = phi_i[k,i,j]
        			if pi > 0.5:
        				fxs.append(fx[k,i,j])
        				fys.append(fy[k,i,j])
        				fzs.append(fz[k,i,j])
    if dims == 1:
    	n, bins, patches = engine.hist(fxs, num_bins,  
                            density = True,  
                            color ='black', 
                            alpha = 0.7, label = 'fx')
    	engine.legend(frameon=False)
    if dims == 2:                     
    	n, bins, patches = engine.hist(fys, num_bins,  
                            density = True,  
                            color ='black', 
                            alpha = 0.7, label = 'fy')
    	engine.legend(frameon=False)
    if dims == 3:        
    	n, bins, patches = engine.hist(fzs, num_bins,  
                            density = True,  
                            color ='black', 
                            alpha = 0.7, label = 'fz')
    	engine.legend(frameon=False)
        
        

def _force(frame, i, v, engine=plt, **kwargs):
    """
    Helper function to plot forces.
    """
    c = frame.com[i]
    engine.arrow(c[1], c[0], v[1], v[0], **kwargs)

def velocity(frame, engine=plt, color='r'):
    """
    Plot total velocity of each cell.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
        color: Color of the arrow.
    """
    scale = frame.parameters['ninfo']*frame.parameters['nsubsteps']
    for i in range(frame.nphases):
        _force(frame, i,
               scale*frame.velocity[i],
               engine=engine,
               color=color)

        
def pressure_force(frame, engine=plt, color='b'):
    """
    Plot pressure force of each cell.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
        color: Color of the arrow.
    """
    scale = frame.parameters['ninfo']*frame.parameters['nsubsteps']
    for i in range(frame.nphases):
        _force(frame, i,
               scale*frame.Fpressure[i],
               engine=engine,
               color=color)  
        
def shape(frame, engine=plt, **kwargs):
    """
    Print shape tensor of each cell as the director of a nematic tensor.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the arrow function.
    """
    _update_dict(kwargs, 'color', 'k')
    for i in range(frame.nphases):
        S00 = frame.S00[i]
        S01 = frame.S01[i]
        S02 = frame.S02[i]
        S12 = frame.S12[i]
        S11 = frame.S11[i]
        S22 = frame.S22[i]
        
        A = np.array([[S00,S01,S02],[S01,S11,S12],[S02,S12,S22]])
        eigvals, eigvecs = LA.eig(A)
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
        engine.arrow(c[1], c[0],  a*nx,  a*ny, **kwargs)
        engine.arrow(c[1], c[0], -a*nx, -a*ny, **kwargs)    
        
        
def cell_shape(frame, i, engine=plt, **kwargs ):

    #_update_dict(kwargs, 'color', 'k')
    #for i in range(frame.nphases):
    S00 = frame.S00[i]
    S01 = frame.S01[i]
    S02 = frame.S02[i]
    S12 = frame.S12[i]
    S11 = frame.S11[i]
    S22 = frame.S22[i]
    A = np.array([[S00,S01,S02],[S01,S11,S12],[S02,S12,S22]])
    eigvals, eigvecs = LA.eig(A)
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
    engine.arrow(c[1], c[0],  a*nx,  a*ny, **kwargs)
    engine.arrow(c[1], c[0], -a*nx, -a*ny, **kwargs)       
    return nx,ny
        
        
def com(frame, engine=plt):
    """
    Plot the center-of-mass of each cell as a red dot. Not really
    photorealistic.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
    """
    for c in frame.com:
        engine.plot(c[1], c[0], 'ro')  
        
        
def label_cell(frame,props,engine=plt):
	cellId = 0
	for c in frame.com:
		label = props[cellId]
		engine.text(c[1],c[0],str(label),fontsize=7)     
		cellId += 1
        
        
def mark_cell(frame,index, engine=plt):
    """
    Plot the center-of-mass of each cell as a red dot. Not really
    photorealistic.

    Args:
        frame: Frame to plot, from archive module.
        engine: Plotting engine or axis.
    """
    c = frame.com
    #for c in frame.com:
    engine.plot(c[index][1], c[index][0], 'k+')  
    
    
    
def mark_cell_outline(frame, zxsec, index, engine=plt ):
	cell(frame, index, zxsec, engine, color='k') 
    
    
        
'''    
def get_corr2(ux, uy, uz, size):
    """
    Compute the correlation (as a function of distance) of two real two-
    dimensional scalar fields.

    Arguments:
        ux, uy: The scalar fields.

    Returns:
        The correlation of ux and uy as an array.
    """
    # get 2d correlations
    cx = np.fft.rfft2(ux)
    cx = np.fft.irfft2(np.multiply(cx, np.conj(cx)))
    cy = np.fft.rfft2(uy)
    cy = np.fft.irfft2(np.multiply(cy, np.conj(cy)))
    cz = np.fft.rfft2(uz)
    cz = np.fft.irfft2(np.multiply(cz, np.conj(cz)))    
    c = cx + cy + cz
    s = size
    r = np.zeros(s)
    n = np.zeros(s)
    k = 0
    for (i, j, m), v in np.ndenumerate(c):
        k = int(sqrt(i**2 + j**2 + m**2))
        if k >= s:
            continue
        r[k] += v
        n[k] += 1
    r = np.divide(r, n)
    r /= r[0]
    return r
'''

def get_corr2d(ux, uy, uz, zxsec):
    """
    Compute the correlation (as a function of distance) of two real two-
    dimensional scalar fields.

    Arguments:
        ux, uy: The scalar fields.

    Returns:
        The correlation of ux and uy as an array.
    """
    # get 2d correlations
    ux = ux[zxsec,:,:]
    uy = uy[zxsec,:,:]
    uz = uz[zxsec,:,:]
    
    cx = np.fft.rfft2(ux)
    cx = np.fft.irfft2(np.multiply(cx, np.conj(cx)))
    cy = np.fft.rfft2(uy)
    cy = np.fft.irfft2(np.multiply(cy, np.conj(cy)))
    cz = np.fft.rfft2(uz)
    cz = np.fft.irfft2(np.multiply(cz, np.conj(cz)))    
    c = cx + cy 
    #s = size
    s = int(sqrt(c.size)/2)

    r = np.zeros(s)
    n = np.zeros(s)
    k = 0
    for (i, j), v in np.ndenumerate(c):
        k = int(sqrt(i**2 + j**2 ))
        if k >= s:
            continue
        r[k] += v
        n[k] += 1
    r = np.divide(r, n)
    r /= r[0]
    return r


def get_corr3d(ux, uy, uz):
    """
    Compute the correlation (as a function of distance) of two real two-
    dimensional scalar fields.

    Arguments:
        ux, uy: The scalar fields.

    Returns:
        The correlation of ux and uy as an array.
    """
    # get 2d correlations
    cx = np.fft.rfft2(ux)
    cx = np.fft.irfft2(np.multiply(cx, np.conj(cx)))
    cy = np.fft.rfft2(uy)
    cy = np.fft.irfft2(np.multiply(cy, np.conj(cy)))
    cz = np.fft.rfft2(uz)
    cz = np.fft.irfft2(np.multiply(cz, np.conj(cz)))    
    c = cx + cy + cz
    #s = size
    s = int(sqrt(c.size)/2)

    r = np.zeros(s)
    n = np.zeros(s)
    k = 0
    for (i, j, m), v in np.ndenumerate(c):
        k = int(sqrt(i**2 + j**2 + m**2))
        if k >= s:
            continue
        r[k] += v
        n[k] += 1
    r = np.divide(r, n)
    r /= r[0]
    return r


def vector_force_field(frame, zxsec, size, avg, minVal,maxVal,engine=plt, quiv=True, cbar=False):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        engine: Plotting engine or axis.
        magn: Plot velocity magnitude as a heatmap?
        cbar: Show color bar?
        avg: Size of the averaging (drops points)
    """
    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.Fpressure, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls'])

    #if magn:
    m = np.sqrt(fx**2 + fy**2 + fz**2)
    print('min = ',np.min(fz[zxsec,:,:]))
    print('max = ',np.max(fz[zxsec,:,:]))

    cax = engine.imshow(fz[zxsec,:,:], interpolation='lanczos', cmap='RdBu',
                            origin='lower',vmin=minVal,vmax=maxVal)
    if cbar:
    	plt.colorbar(cax,shrink=0.3)
    	#plt.colorbar(cax,orientation='horizontal',extend='min')

    fx = fx[zxsec,:,:];
    fy = fy[zxsec,:,:];

    fx = fx.reshape((fx.shape[0]//avg, avg, fx.shape[1]//avg, avg))
    fx = np.mean(fx, axis=(1, 3))
    fy = fy.reshape((fy.shape[0]//avg, avg, fy.shape[1]//avg, avg))
    fy = np.mean(fy, axis=(1, 3))

    if quiv:
    	cax = engine.quiver(np.arange(0, frame.parameters['Size'][1], step=avg),
                        np.arange(0, frame.parameters['Size'][0], step=avg),
                        fy, fx,
                        pivot='tail', units='dots', scale_units='dots', color='k')  


def extract_ector_force_field(frame, size, avg):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.Fpressure, size, mode=mode)
    
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls'])
    
    return fx,fy,fz


    '''
    fx = fx.reshape((fx.shape[0]//avg, avg, fx.shape[1]//avg, avg))
    fx = np.mean(fx, axis=(1, 3))
    
    fy = fy.reshape((fy.shape[0]//avg, avg, fy.shape[1]//avg, avg))
    fy = np.mean(fy, axis=(1, 3))

    fz = fz.reshape((fz.shape[0]//avg, avg, fy.shape[1]//avg, avg))
    fz = np.mean(fz, axis=(1, 3))
    ''' 
     
     
def velocity_xsection(fig,frame, yxsec, size,engine=plt):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    vx, vy, vz = get_velocity_field(frame.phi, frame.velocity, size, mode=mode)
    vx *= (1.-frame.parameters['walls'])
    vy *= (1.-frame.parameters['walls'])
    vz *= (1.-frame.parameters['walls'])
    m = np.sqrt(vx**2 + vy**2 + vz**2)
    
    (LZ, LX, LY) = vx.shape
    xsection = np.zeros((LZ,LX))
    for k in range(LZ):
    	for i in range(LX):
    		for j in range(LY):
    			if (j == yxsec):
    				xsection[k,i] = vz[k,i,j]
    				
    				
    minVal = np.min(vz.ravel())
    maxVal = np.max(vz.ravel())	
    minVal = -0.03
    maxVal = 0.1
    cmap = mpl.cm.RdBu
    normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)	
    print(minVal,0,maxVal)
    caxx = engine.imshow(xsection, interpolation='lanczos', cmap=cmap,norm=normi,origin='lower',vmin=minVal,vmax=maxVal)
    cbar_ax = fig.add_axes([0.4, 0.36, 0.2, 0.02]) #left bottom width height 
    cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    cbar.ax.set_title('velocity, $v_{z}$', fontsize=20)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()



              
def layer_cross_section(fig,frame,lprop,yxsec,engine=plt):

		
	(LZ,LX,LY) = frame.phi[0].shape
	field = np.zeros((LZ,LX,LY))
	for i in range(len(frame.phi)):
		p = frame.phi[i]
		if(lprop[i] == 1):
			field += p * -1.;
		if(lprop[i] == 2):
			field += p * 1.;
		p = p[:,:,yxsec]
		
	xsection = np.zeros((LZ,LX))
	for k in range(LZ):
		for i in range(LX):
			for j in range(LY):
				if (j == yxsec):
					xsection[k,i] = field[k,i,j]
	
	minVal = np.min(xsection.ravel())
	maxVal = np.max(xsection.ravel())	
	#minVal = -0.03
	#maxVal = 0.1
	cmap = mpl.cm.RdBu
	normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)	
	caxx = engine.imshow(xsection, interpolation='lanczos', cmap=cmap,norm=normi,origin='lower',vmin=minVal,vmax=maxVal)
	cbar_ax = fig.add_axes([0.4, 0.36, 0.2, 0.02]) #left bottom width height 
	cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
	cbar.ax.set_title('velocity, $v_{z}$', fontsize=20)
	cbar.formatter.set_powerlimits((0, 0))
	cbar.update_ticks()

	'''	
	for i in range(len(frame.phi)):
		p = frame.phi[i]
		p = p[:,:,yxsec]
		
		unqprops = np.unique(lprop)
		number_of_bins = len(unqprops)
		color = ['#0047AB','#6E260E','#000000','#6f03fc','#FFC300','#FF5733','#bf0823','#581845']
		ncells = len(frame.phi)
		#tprop = lprop[fr*ncells:(fr+1)*ncells:1];
		map_value = np.where(lprop[i]==unqprops)[0]
		map_value = map_value[0]
		map_color = color[int(map_value)]
		
		engine.contourf(np.arange(0, frame.parameters['Size'][1]),
                   np.arange(0, frame.parameters['Size'][2]),
                   p, linewidths =[0.5],colors=map_color,levels=[0.5,1.5],alpha=0.8)
	'''	
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    


    

        
        
        
        
        
        
        
        

            

def press_field(frame, zxsec, size, minVal,maxVal,engine=plt, cbar=False):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    sxx = frame.stress_xx
    sxx = get_stress_field(sxx, size, mode='wrap')
    sxx *= (1.-frame.parameters['walls'])
    
    cax = engine.imshow(sxx[zxsec,:,:], interpolation='lanczos', cmap='RdBu', origin = 'lower',vmin=minVal,vmax=maxVal)
    print('min = ',np.min(sxx[zxsec,:,:]))
    print('max = ',np.max(sxx[zxsec,:,:]))


def bilin_interp(k,i,j,pvec):	
	#print(k,' ',i,' ',j)
	z0 = math.floor(k)
	x0 = math.floor(i)
	y0 = math.floor(j)
	z1 = math.ceil(k)
	x1 = math.ceil(i)
	y1 = math.ceil(j)
	
	#p000 = getGridValue(pvec,x0,y0,z0)
	#p001 = getGridValue(pvec,x0,y0,z1)
	#p010 = getGridValue(pvec,x0,y1,z0)
	#p011 = getGridValue(pvec,x0,y1,z1)
	#p101 = getGridValue(pvec,x1,y0,z1)
	
	p000 = pvec[z0,x0,y0]
	p001 = pvec[z1,x0,y0]
	p010 = pvec[z0,x0,y1]
	p011 = pvec[z1,x0,y1]
	p101 = pvec[z1,x1,y0]
	
	#xm = getGridValue(pvec,x1,y0,z0) - p000
	xm = pvec[z0,x1,y0] - p000
	ym = p010 - p000
	zm = p001 - p000
	#xym	= -xm - p010 + getGridValue(pvec,x1,y1,z0);
	xym	= -xm - p010 + pvec[z0,x1,y1]
	xzm	= -xm - p001 + p101;
	yzm	= -ym - p001 + p011;
	#xyzm = -xym + p001 - p101 - p011 + getGridValue(pvec,x1,y1,z1);
	xyzm = -xym + p001 - p101 - p011 + pvec[ z1,x1,y1 ]
	dx 	= i - (x0);
	dy 	= j - (y0);
	dz	= k - (z0);
	value = p000 + xm*dx + ym*dy + zm*dz + xym*dx*dy + xzm*dx*dz + yzm*dy*dz + xyzm*dx*dy*dz;
	return value 
	
def get_solid_stress(frame,sig,LZ0,LX0,LY0):

	(nz,nx,ny) = sig.shape
	
	x = np.linspace(0, LX0-1, nx)
	y = np.linspace(0, LY0-1, ny)
	z = np.linspace(0, LZ0-1, nz)
	
	phi_tot = np.zeros((LZ0,LX0,LY0))
	
	for ii in range(len(frame.phi)):
		phi_tot += frame.phi[ii]
	
	sigs = []
	for k in range(nz):
		for i in range(nx):
			for j in range(ny):
				pi = bilin_interp(z[k],x[i],y[j],phi_tot)
				#pi = phi_i[k,i,j]
				if pi > 0.5:
					sigs.append(sig[k,i,j])
	return sigs
	
           
           
   
            

def cg_data(cgL,LZ,LY,LX,data):

	nvx = int(LX / cgL + 1)
	nvy = int(LY / cgL + 1)
	nvz = int(LZ / cgL + 1)
	nsites = int(cgL)
	
	#mcgVol = np.zeros( (nvx-1)*(nvy-1)*(nvz-1))
	#cgVolX = np.zeros( (nvx-1)*(nvy-1)*(nvz-1))
	#cgVolY = np.zeros( (nvx-1)*(nvy-1)*(nvz-1))
	#cgVolZ = np.zeros( (nvx-1)*(nvy-1)*(nvz-1))
	intX = np.zeros(nsites)
	intY = np.zeros(nsites)
	intZ = np.zeros(nsites)
	cgVol = np.zeros((nsites,nsites,nsites))
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	for i in range(nvx):
		LengthX[i] = i * (nsites-1)
	for i in range(nvy):
		LengthY[i] = i * (nsites-1)
	for i in range(nvz):
		LengthZ[i] = i * (nsites-1)
		
	vol = 0	
	for z in range(nvz-1):
		lbz = LengthZ[z]
		ubz = LengthZ[z+1]
		for x in range(nvx-1):
			lbx = LengthX[x]
			ubx = LengthX[x+1]
			for y in range(nvy-1):
				lby = LengthY[y]
				uby = LengthY[y+1]
				
				for i in range(nsites):
					intX[i] = lbx + i
					intY[i] = lby + i
					intZ[i] = lbz + i
					
				for zz in range(nsites):
					for xx in range(nsites):
						for yy in range(nsites):
							cgVol[zz,xx,yy] = data[int(intZ[zz]),int(intX[xx]),int(intY[yy])]
								
				#mcgVol[vol] = np.mean(cgVol)
				#cgVolX[vol] = np.mean(intX)
				#cgVolY[vol] = np.mean(intY)
				#cgVolZ[vol] = np.mean(intZ)
                

			
def cut_wall_for_cg(LZ,LY,LX,wall_thickness,data):
    		
	resize_data = np.zeros((int(LZ - wall_thickness),LX,LY))
	
	for z in range(LZ):
		for x in range(LX):
			for y in range(LY):
				if (z > wall_thickness):
					zn = int(z - wall_thickness)
					resize_data[zn,x,y] = data[z,x,y] 
					
	return resize_data  

    
def resize_data_for_cg(cgL,LZ,LY,LX,data):
    
	lxp = cgL - (LX%cgL)
	lyp = cgL - (LY%cgL)
	lzp = cgL - (LZ%cgL)
	
	LXn = LX
	LYn = LY
	LZn = LZ
	
	if ( (LX%cgL) > 0):
		LXn = LX + lxp
	if ( (LY%cgL) > 0):
		LYn = LY + lyp
	if ( (LZ%cgL) > 0):
		LZn = LZ + lzp
		
	resize_data = np.zeros((LZn,LXn,LYn))
	
	for z in range(LZn):
		for x in range(LXn):
			for y in range(LYn):
				if (z < LZ):
					zn = z
				if (x < LX):
					xn = x
				if (y < LY):
					yn = y
				if (z >= LZ):
					zn = z - LZ
				if (y >= LY):
					yn = y - LY
				if (x >= LX):
					xn = x - LX
					
				resize_data[z,x,y] = data[zn,xn,yn] 
					
	return resize_data     
    
    
    
def resize_data_for_cg_3d(cgLz,cgLy,cgLx,LZ,LY,LX,data):
    
	lxp = cgLx - (LX%cgLx)
	lyp = cgLy - (LY%cgLy)
	lzp = cgLz - (LZ%cgLz)
	
	LXn = LX
	LYn = LY
	LZn = LZ
	
	if ( (LX%cgLx) > 0):
		LXn = LX + lxp
	if ( (LY%cgLy) > 0):
		LYn = LY + lyp
	if ( (LZ%cgLz) > 0):
		LZn = LZ + lzp
		
	resize_data = np.zeros((LZn,LXn,LYn))
	
	for z in range(LZn):
		for x in range(LXn):
			for y in range(LYn):
				if (z < LZ):
					zn = z
				if (x < LX):
					xn = x
				if (y < LY):
					yn = y
				if (z >= LZ):
					zn = z - LZ
				if (y >= LY):
					yn = y - LY
				if (x >= LX):
					xn = x - LX
					
				resize_data[z,x,y] = data[zn,xn,yn] 
					
	return resize_data     
	
	
    
def discrete_stresses(cgL,LZ,LY,LX,fx,fy,fz,zxsec,wall_thick, engine=plt, **kwargs):
	
	nvx = int(LX / cgL + 1)
	nvy = int(LY / cgL + 1)
	nvz = int(LZ / cgL + 1)
	nsites = int(cgL)

	coorX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	coorY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	coorZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
    
	nx = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	ny = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	S = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	cX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	cY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	cZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
    
	sigXX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigXY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigXZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigYX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigYY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigYZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigZX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigZY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigZZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
    
	intX = np.zeros(nsites)
	intY = np.zeros(nsites)
	intZ = np.zeros(nsites)
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	for i in range(nvx):
		LengthX[i] = i * (nsites)
	for i in range(nvy):
		LengthY[i] = i * (nsites)
	for i in range(nvz):
		LengthZ[i] = i * (nsites)
		
	vol = 0	
	for z in range(nvz-1):
		lbz = LengthZ[z]
		ubz = LengthZ[z+1]
		for x in range(nvx-1):
			lbx = LengthX[x]
			ubx = LengthX[x+1]
			for y in range(nvy-1):
				lby = LengthY[y]
				uby = LengthY[y+1]
				
				for i in range(nsites):
					intX[i] = lbx + i
					intY[i] = lby + i
					intZ[i] = lbz + i
                   
				cc = np.array([np.mean(intX), np.mean(intY), np.mean(intZ)]) 
				coorX[z,x,y] = cc[0]
				coorY[z,x,y] = cc[1]
				coorZ[z,x,y] = cc[2]
				sxx = 0
				sxy = 0
				sxz = 0
				syx = 0
				syy = 0
				syz = 0
				szx = 0
				szy = 0
				szz = 0
				for zz in range(nsites):
					for xx in range(nsites):
					    for yy in range(nsites):
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

				sxx = sxx/(nsites*nsites*nsites)
				sxy = sxy/(2.*nsites*nsites*nsites)  
				sxz = sxz/(2.*nsites*nsites*nsites)  
				syx = syx/(2.*nsites*nsites*nsites)  
				syy = syy/(nsites*nsites*nsites)  
				syz = syz/(2.*nsites*nsites*nsites)  
				szx = szx/(2.*nsites*nsites*nsites)  
				szy = szy/(2.*nsites*nsites*nsites)  
				szz = szz/(nsites*nsites*nsites)  
                
				sigXX[vol] = sxx    
				sigXY[vol] = sxy  
				sigXZ[vol] = sxz  
				sigYX[vol] = syx  
				sigYY[vol] = syy  
				sigYZ[vol] = syz  
				sigZX[vol] = szx  
				sigZY[vol] = szy  
				sigZZ[vol] = szz  
				cX[vol] = cc[0]  
				cY[vol] = cc[1]  
				cZ[vol] = cc[2] 

				vol = vol + 1

				sTen =  np.array([[sxx,sxy,sxz],[syx,syy,syz],[szx,szy,szz]])
				eigvals, eigvecs = LA.eig(sTen)
				idx = eigvals.argsort()[::-1]   
				eigvals = eigvals[idx] 
				eigvecs = eigvecs[:,idx]
				maxIndex = np.where(eigvals == np.amax(eigvals))
				maxEigVec = eigvecs[:,maxIndex].ravel()
				norm = np.linalg.norm(maxEigVec)
                
				nx[z,x,y] = maxEigVec[0]
				ny[z,x,y] = maxEigVec[1]
				S[z,x,y] = norm

	nx = nx[zxsec,:,:] 
	ny = ny[zxsec,:,:] 
	coorX = coorX[zxsec,:,:] 
	coorY = coorY[zxsec,:,:] 
	S = np.vectorize(sqrt)(nx**2 + ny**2) 

	x = [] 
	y = [] 
	#np.savetxt('cgStresses.dat',np.c_[cX,cY,cZ,sigXX,sigXY,sigXZ,sigYX,sigYY,sigYZ,sigZX,sigZY,sigZZ])

	for i, j in product(np.arange( (nvx-1), step=1),
                        np.arange( (nvy-1), step=1)):

		f = cgL 
		x.append(coorX[i,j] + .5 - f*nx[i, j]/2.)
		x.append(coorX[i,j] + .5 + f*nx[i, j]/2.)
		x.append(None)
		y.append(coorY[i,j] + .5 - f*ny[i, j]/2.)
		y.append(coorY[i,j] + .5 + f*ny[i, j]/2.)
		y.append(None)
		#x.append(i + .5 - f*nx[i, j]/2.)
		#x.append(i + .5 + f*nx[i, j]/2.)
		#x.append(None)
		#y.append(j + .5 - f*ny[i, j]/2.)
		#y.append(j + .5 + f*ny[i, j]/2.)
		#y.append(None)
		#engine.plot(coorY[i,j],coorX[i,j],'ro', markerfacecolor="None",markersize=2)
		#engine.plot((cgL/2.)+(j)*cgL,(cgL/2.)+(i)*cgL, 'b.', markerfacecolor="None",markersize=2)

	_update_dict(kwargs, 'linestyle', '-')
	_update_dict(kwargs, 'linewidth', 0.45)
	engine.plot(y, x,color='k',**kwargs)  

	return nx,ny


def stress_defects_noplot(cgL,LZ,LY,LX,fx,fy,fz,zxsec,wall_thick, engine=plt, **kwargs):
	
	nvx = int(LX / cgL + 1)
	nvy = int(LY / cgL + 1)
	nvz = int(LZ / cgL + 1)
	nsites = int(cgL)

	coorX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	coorY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	coorZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
    
	nx = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	ny = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	S = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	cX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	cY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	cZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
    
	sigXX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigXY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigXZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigYX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigYY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigYZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigZX = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigZY = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
	sigZZ = np.zeros(((nvz-1)*(nvx-1)*(nvy-1)))
    
	intX = np.zeros(nsites)
	intY = np.zeros(nsites)
	intZ = np.zeros(nsites)
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	for i in range(nvx):
		LengthX[i] = i * (nsites)
	for i in range(nvy):
		LengthY[i] = i * (nsites)
	for i in range(nvz):
		LengthZ[i] = i * (nsites)
		
	vol = 0	
	for z in range(nvz-1):
		lbz = LengthZ[z]
		ubz = LengthZ[z+1]
		for x in range(nvx-1):
			lbx = LengthX[x]
			ubx = LengthX[x+1]
			for y in range(nvy-1):
				lby = LengthY[y]
				uby = LengthY[y+1]
				
				for i in range(nsites):
					intX[i] = lbx + i
					intY[i] = lby + i
					intZ[i] = lbz + i
                   
				cc = np.array([np.mean(intX), np.mean(intY), np.mean(intZ)]) 
				coorX[z,x,y] = cc[0]
				coorY[z,x,y] = cc[1]
				coorZ[z,x,y] = cc[2]
				sxx = 0
				sxy = 0
				sxz = 0
				syx = 0
				syy = 0
				syz = 0
				szx = 0
				szy = 0
				szz = 0
				for zz in range(nsites):
					for xx in range(nsites):
					    for yy in range(nsites):
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

				sxx = sxx/(nsites*nsites*nsites)
				sxy = sxy/(2.*nsites*nsites*nsites)  
				sxz = sxz/(2.*nsites*nsites*nsites)  
				syx = syx/(2.*nsites*nsites*nsites)  
				syy = syy/(nsites*nsites*nsites)  
				syz = syz/(2.*nsites*nsites*nsites)  
				szx = szx/(2.*nsites*nsites*nsites)  
				szy = szy/(2.*nsites*nsites*nsites)  
				szz = szz/(nsites*nsites*nsites)  
                
				sigXX[vol] = sxx    
				sigXY[vol] = sxy  
				sigXZ[vol] = sxz  
				sigYX[vol] = syx  
				sigYY[vol] = syy  
				sigYZ[vol] = syz  
				sigZX[vol] = szx  
				sigZY[vol] = szy  
				sigZZ[vol] = szz  
				cX[vol] = cc[0]  
				cY[vol] = cc[1]  
				cZ[vol] = cc[2] 

				vol = vol + 1

				sTen =  np.array([[sxx,sxy,sxz],[syx,syy,syz],[szx,szy,szz]])
				eigvals, eigvecs = LA.eig(sTen)
				idx = eigvals.argsort()[::-1]   
				eigvals = eigvals[idx] 
				eigvecs = eigvecs[:,idx]
				maxIndex = np.where(eigvals == np.amax(eigvals))
				maxEigVec = eigvecs[:,maxIndex].ravel()
				norm = np.linalg.norm(maxEigVec)
                
				nx[z,x,y] = maxEigVec[0]
				ny[z,x,y] = maxEigVec[1]
				S[z,x,y] = norm

	nx = nx[zxsec,:,:] 
	ny = ny[zxsec,:,:] 

	return nx,ny


def get_sdirector_fields(cgL,LZ,LY,LX,fx,fy,fz):
	
	nvx = int(LX / cgL + 1)
	nvy = int(LY / cgL + 1)
	nvz = int(LZ / cgL + 1)
	nsites = int(cgL)
    
	nx = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	ny = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	nz = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
    
	intX = np.zeros(nsites)
	intY = np.zeros(nsites)
	intZ = np.zeros(nsites)
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	for i in range(nvx):
		LengthX[i] = i * (nsites)
	for i in range(nvy):
		LengthY[i] = i * (nsites)
	for i in range(nvz):
		LengthZ[i] = i * (nsites)
		
	vol = 0	
	for z in range(nvz-1):
		lbz = LengthZ[z]
		ubz = LengthZ[z+1]
		for x in range(nvx-1):
			lbx = LengthX[x]
			ubx = LengthX[x+1]
			for y in range(nvy-1):
				lby = LengthY[y]
				uby = LengthY[y+1]
				
				for i in range(nsites):
					intX[i] = lbx + i
					intY[i] = lby + i
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
				for zz in range(nsites):
					for xx in range(nsites):
					    for yy in range(nsites):
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

				sxx = sxx/(nsites*nsites*nsites)
				sxy = sxy/(2.*nsites*nsites*nsites)  
				sxz = sxz/(2.*nsites*nsites*nsites)  
				syx = syx/(2.*nsites*nsites*nsites)  
				syy = syy/(nsites*nsites*nsites)  
				syz = syz/(2.*nsites*nsites*nsites)  
				szx = szx/(2.*nsites*nsites*nsites)  
				szy = szy/(2.*nsites*nsites*nsites)  
				szz = szz/(nsites*nsites*nsites)  
                
				vol = vol + 1

				sTen =  np.array([[sxx,sxy,sxz],[syx,syy,syz],[szx,szy,szz]])
				eigvals, eigvecs = LA.eig(sTen)
				idx = eigvals.argsort()[::-1]   
				eigvals = eigvals[idx] 
				eigvecs = eigvecs[:,idx]
				maxIndex = np.where(eigvals == np.amax(eigvals))
				maxEigVec = eigvecs[:,maxIndex].ravel()
				norm = np.linalg.norm(maxEigVec)
                
				nx[z,x,y] = maxEigVec[0]
				ny[z,x,y] = maxEigVec[1]
				nz[z,x,y] = maxEigVec[2]

	return nx,ny,nz 
	
def get_stress_Tensor(cgL,LZ,LY,LX,fx,fy,fz):
	
	nvx = int(LX / cgL + 1)
	nvy = int(LY / cgL + 1)
	nvz = int(LZ / cgL + 1)
	nsites = int(cgL)
    
	sigXX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigXY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigXZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigYX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigYY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigYZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigZX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigZY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigZZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
    
	intX = np.zeros(nsites)
	intY = np.zeros(nsites)
	intZ = np.zeros(nsites)
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	for i in range(nvx):
		LengthX[i] = i * (nsites)
	for i in range(nvy):
		LengthY[i] = i * (nsites)
	for i in range(nvz):
		LengthZ[i] = i * (nsites)
		
	vol = 0	
	for z in range(nvz-1):
		lbz = LengthZ[z]
		ubz = LengthZ[z+1]
		for x in range(nvx-1):
			lbx = LengthX[x]
			ubx = LengthX[x+1]
			for y in range(nvy-1):
				lby = LengthY[y]
				uby = LengthY[y+1]
				
				for i in range(nsites):
					intX[i] = lbx + i
					intY[i] = lby + i
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
				for zz in range(nsites):
					for xx in range(nsites):
					    for yy in range(nsites):
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

				sxx = sxx/(nsites*nsites*nsites)
				sxy = sxy/(2.*nsites*nsites*nsites)  
				sxz = sxz/(2.*nsites*nsites*nsites)  
				syx = syx/(2.*nsites*nsites*nsites)  
				syy = syy/(nsites*nsites*nsites)  
				syz = syz/(2.*nsites*nsites*nsites)  
				szx = szx/(2.*nsites*nsites*nsites)  
				szy = szy/(2.*nsites*nsites*nsites)  
				szz = szz/(nsites*nsites*nsites)  
                
				sigXX[z,x,y] = sxx    
				sigXY[z,x,y] = sxy  
				sigXZ[z,x,y] = sxz   
				sigYY[z,x,y] = syy  
				sigYZ[z,x,y] = syz  
				sigZZ[z,x,y] = szz  


				vol = vol + 1


	return sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ

	
	
	
	
	









	





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
	
	vol = 0	
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
				#fig = plt.figure()
				#ax = plt.axes(projection='3d')
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

				vol = vol + 1

	xsp = np.linspace(0, LX-1, (nvx-1))
	ysp = np.linspace(0, LY-1, (nvy-1))
	zsp = np.linspace(0, LZ-1, (nvz-1))
    	
	return sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ,xsp,ysp,zsp

	
	
	
	
		
	
	
	
	
	
	
	
	
	
	
	
	
	

def coarse_grain_stress_field_3d(cgLz,cgLy,cgLx,LZ,LY,LX,fx,fy,fz):
	
	nvx = int(LX / cgLx + 1)
	nvy = int(LY / cgLy + 1)
	nvz = int(LZ / cgLz + 1)
	
	nsitesX = int(cgLx)
	nsitesY = int(cgLy)
	nsitesZ = int(cgLz)
    
	sigXX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigXY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigXZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigYX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigYY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigYZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigZX = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigZY = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
	sigZZ = np.zeros(((nvz-1),(nvx-1),(nvy-1)))
    
	intX = np.zeros(nsitesX)
	intY = np.zeros(nsitesY)
	intZ = np.zeros(nsitesZ)
	LengthX = np.zeros(nvx)
	LengthY = np.zeros(nvy)
	LengthZ = np.zeros(nvz)
	
	for i in range(nvx):
		LengthX[i] = i * (nsitesX)
	for i in range(nvy):
		LengthY[i] = i * (nsitesY)
	for i in range(nvz):
		LengthZ[i] = i * (nsitesZ)
		
	vol = 0	
	for z in range(nvz-1):
		lbz = LengthZ[z]
		ubz = LengthZ[z+1]
		for x in range(nvx-1):
			lbx = LengthX[x]
			ubx = LengthX[x+1]
			for y in range(nvy-1):
				lby = LengthY[y]
				uby = LengthY[y+1]
				
				for i in range(nsitesX):
					intX[i] = lbx + i
				for i in range(nsitesY):
					intY[i] = lby + i
				for i in range(nsitesZ):
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
				for zz in range(nsitesZ):
					for xx in range(nsitesY):
					    for yy in range(nsitesX):
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

				sxx = sxx/(nsitesX*nsitesY*nsitesZ)
				sxy = sxy/(2.*nsitesX*nsitesY*nsitesZ)
				sxz = sxz/(2.*nsitesX*nsitesY*nsitesZ)
				syx = syx/(2.*nsitesX*nsitesY*nsitesZ)
				syy = syy/(nsitesX*nsitesY*nsitesZ)
				syz = syz/(2.*nsitesX*nsitesY*nsitesZ)
				szx = szx/(2.*nsitesX*nsitesY*nsitesZ)
				szy = szy/(2.*nsitesX*nsitesY*nsitesZ)
				szz = szz/(nsitesX*nsitesY*nsitesZ)
                
				sigXX[z,x,y] = sxx    
				sigXY[z,x,y] = sxy  
				sigXZ[z,x,y] = sxz   
				sigYY[z,x,y] = syy  
				sigYZ[z,x,y] = syz  
				sigZZ[z,x,y] = szz  

				vol = vol + 1

	xsp = np.linspace(0, LX-1, (nvx-1))
	ysp = np.linspace(0, LY-1, (nvy-1))
	zsp = np.linspace(0, LZ-1, (nvz-1))
    	#for z in (zsp):
    	#	for y in (ysp):
    	#		for x in (xsp):
    	
	return sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ,xsp,ysp,zsp

	
	
def coarse_grain_stress_field(f1,cgLz,cgLy,cgLx,size,HiRes_cg = True,engine=plt):

    mode = 'wrap' if f1.parameters['BC'] == 0 else 'constant'
    f1x, f1y, f1z = get_force_field(f1.phi, f1.parameters['xi']*f1.velocity-f1.Fpol, size, mode=mode)
    f1x *= (1.-f1.parameters['walls'])
    f1y *= (1.-f1.parameters['walls'])
    f1z *= (1.-f1.parameters['walls']) 
    (LZ, LX, LY) = f1x.shape

    cut_fx = cut_wall_for_cg(LZ,LY,LX,f1.parameters['wall_thickness'],f1x)
    cut_fy = cut_wall_for_cg(LZ,LY,LX,f1.parameters['wall_thickness'],f1y)
    cut_fz = cut_wall_for_cg(LZ,LY,LX,f1.parameters['wall_thickness'],f1z)
    (LZ, LX, LY) = cut_fx.shape

    resized_fx = resize_data_for_cg_3d(cgLz,cgLy,cgLx,LZ,LY,LX,cut_fx)
    resized_fy = resize_data_for_cg_3d(cgLz,cgLy,cgLx,LZ,LY,LX,cut_fy)
    resized_fz = resize_data_for_cg_3d(cgLz,cgLy,cgLx,LZ,LY,LX,cut_fz)
    (LZn, LXn, LYn) = resized_fx.shape

    if HiRes_cg:
    	sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ,xsp,ysp,zsp = coarse_grain_stress_field_3d_07052022(cgLz,cgLy,cgLx,LZ,LY,LX,resized_fx,resized_fy,resized_fz)
    else:
    	sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ,xsp,ysp,zsp = coarse_grain_stress_field_3d(cgLz,cgLy,cgLx,LZ,LY,LX,resized_fx,resized_fy,resized_fz)
    
    return sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ,xsp,ysp,zsp

    
    
    
    
            
def traction2stress(frame, cgL, zxsec, size, engine=plt,show_def=False, **kwargs):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.parameters['xi']*frame.velocity-frame.Fpol, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls']) 
    (LZ, LX, LY) = fx.shape
    #print('shape before',fx.shape)
    cut_fx = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fx)
    cut_fy = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fy)
    cut_fz = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fz)
    (LZ, LX, LY) = cut_fx.shape
    #print('shape after wall remvoal', cut_fx.shape)
    resized_fx = resize_data_for_cg(cgL,LZ,LY,LX,cut_fx)
    resized_fy = resize_data_for_cg(cgL,LZ,LY,LX,cut_fy)
    resized_fz = resize_data_for_cg(cgL,LZ,LY,LX,cut_fz)
    (LZn, LXn, LYn) = resized_fx.shape
    #print('shape after resize', resized_fx.shape)
    dx,dy = discrete_stresses(cgL,LZn,LYn,LXn,resized_fx,resized_fy,resized_fz,zxsec,frame.parameters['wall_thickness'],engine=engine, **kwargs)
    w = charge_array(dx,dy)
    defects = get_defects(w)
    for d in defects:
    	if d['charge'] == 0.5:
    		engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'b<', markerfacecolor="None",markersize=2)
    	elif d['charge'] == -0.5:
    		engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'g>', markerfacecolor="None",markersize=2)
    	elif d['charge'] == 1.:
    		engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'k*', markerfacecolor="None",markersize=2)
    	elif d['charge'] == -1.:
    		engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'ys', markerfacecolor="None",markersize=2)
      

def stress_field_defects(frame, cgL, zxsec, size, engine=plt, **kwargs):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.parameters['xi']*frame.velocity-frame.Fpol, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls']) 
    (LZ, LX, LY) = fx.shape
    #print('shape before',fx.shape)
    cut_fx = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fx)
    cut_fy = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fy)
    cut_fz = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fz)
    (LZ, LX, LY) = cut_fx.shape
    #print('shape after wall remvoal', cut_fx.shape)
    resized_fx = resize_data_for_cg(cgL,LZ,LY,LX,cut_fx)
    resized_fy = resize_data_for_cg(cgL,LZ,LY,LX,cut_fy)
    resized_fz = resize_data_for_cg(cgL,LZ,LY,LX,cut_fz)
    (LZn, LXn, LYn) = resized_fx.shape
    #print('shape after resize', resized_fx.shape)
    dx,dy = stress_defects_noplot(cgL,LZn,LYn,LXn,resized_fx,resized_fy,resized_fz,zxsec,frame.parameters['wall_thickness'],engine=engine, **kwargs)
    if show_def:
        w = charge_array(dx,dy)
        defects = get_defects(w)
        for d in defects:
            if d['charge'] == 0.5:
                engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'bo', markerfacecolor="None",markersize=2)
            elif d['charge'] == -0.5:
                engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'g^', markerfacecolor="None",markersize=2)
            elif d['charge'] == 1.:
                engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'k>', markerfacecolor="None",markersize=2)
            elif d['charge'] == -1.:
                engine.plot( (d["pos"][0])*cgL-(cgL/1.), (d["pos"][1])*cgL-(cgL/1.), 'ys', markerfacecolor="None",markersize=2)

            

def get_stress_directors(frame, cgL,size):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.parameters['xi']*frame.velocity-frame.Fpol, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls']) 
    (LZ, LX, LY) = fx.shape
    #print('shape before',fx.shape)
    cut_fx = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fx)
    cut_fy = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fy)
    cut_fz = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fz)
    (LZ, LX, LY) = cut_fx.shape
    #print('shape after wall remvoal', cut_fx.shape)
    resized_fx = resize_data_for_cg(cgL,LZ,LY,LX,cut_fx)
    resized_fy = resize_data_for_cg(cgL,LZ,LY,LX,cut_fy)
    resized_fz = resize_data_for_cg(cgL,LZ,LY,LX,cut_fz)
    (LZn, LXn, LYn) = resized_fx.shape
    #print('shape after resize', resized_fx.shape)
    dx,dy,dz= get_sdirector_fields(cgL,LZn,LYn,LXn,resized_fx,resized_fy,resized_fz)
    return dx,dy,dz


def director(Q00, Q01, Q02, Q11, Q12, Q22, zxsec, avg=1, scale=False, engine=plt, **kwargs):
    """
    Plot director field associated with a given nematic field.

    Args:
        Qxx, Qxy: Components of the nematic field.
        avg: Coarse-graining size.
        scale: Scale factor that controls the size of the director.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the plot function.
    """

    (LZ, LX, LY) = Q00.shape

    nx = np.zeros((LZ,LX,LY))
    ny = np.zeros((LZ,LX,LY))
    S  = np.zeros((LZ,LX,LY))
    
    for k in range(LZ):
        for i in range(LX):
            for j in range(LY):
                q00 = Q00[k,i,j]
                q01 = Q01[k,i,j]
                q02 = Q02[k,i,j]
                q11 = Q11[k,i,j]
                q12 = Q12[k,i,j]
                q22 = Q22[k,i,j]
                A = np.array([[q00,q01,q02],[q01,q11,q12],[q02,q12,q22]])

                eigvals, eigvecs = LA.eig(A)
                idx = eigvals.argsort()[::-1]   
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:,idx]
                maxIndex = np.where(eigvals == np.amax(eigvals))
                maxEigVec = eigvecs[:,maxIndex].ravel()
                norm = np.linalg.norm(maxEigVec)

                nx[k,i,j] = maxEigVec[0]
                ny[k,i,j] = maxEigVec[1]     

                S[k,i,j] = norm
                
                
    nx = nx[zxsec,:,:]
    ny = ny[zxsec,:,:]
    S  = np.vectorize(sqrt)(nx**2 + ny**2)
    
    #for defects 
    nxd = nx
    nyd = ny
    # coarse grain
    S = ndimage.generic_filter(S, np.mean, size=avg)
    nx = ndimage.generic_filter(nx, np.mean, size=avg)
    ny = ndimage.generic_filter(ny, np.mean, size=avg)



    (LX, LY) = S.shape

    ## construct nematic lines
    x = []
    y = []
    for i, j in product(np.arange(LX, step=avg),
                        np.arange(LY, step=avg)):
        f = avg*(S[i, j] if scale else 1.)
        x.append(i + .5 - f*nx[i, j]/2.)
        x.append(i + .5 + f*nx[i, j]/2.)
        x.append(None)
        y.append(j + .5 - f*ny[i, j]/2.)
        y.append(j + .5 + f*ny[i, j]/2.)
        y.append(None)

        
    _update_dict(kwargs, 'linestyle', '-')
    _update_dict(kwargs, 'linewidth', 0.45)
    engine.plot(y, x,color='r', **kwargs)
    #return nxd,nyd
    return nx,ny

'''
def director_cg(Q00, Q01, Q02, Q11, Q12, Q22, zxsec, avg=1, scale=False, engine=plt, **kwargs):
    """
    Plot director field associated with a given nematic field.

    Args:
        Qxx, Qxy: Components of the nematic field.
        avg: Coarse-graining size.
        scale: Scale factor that controls the size of the director.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the plot function.
    """

    (LZ, LX, LY) = Q00.shape

    nx = np.zeros((LZ,LX,LY))
    ny = np.zeros((LZ,LX,LY))
    S  = np.zeros((LZ,LX,LY))
    
    for k in range(LZ):
        for i in range(LX):
            for j in range(LY):
                q00 = Q00[k,i,j]
                q01 = Q01[k,i,j]
                q02 = Q02[k,i,j]
                q11 = Q11[k,i,j]
                q12 = Q12[k,i,j]
                q22 = Q22[k,i,j]
                A = np.array([[q00,q01,q02],[q01,q11,q12],[q02,q12,q22]])

                eigvals, eigvecs = LA.eig(A)
                idx = eigvals.argsort()[::-1]   
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:,idx]
                maxIndex = np.where(eigvals == np.amax(eigvals))
                maxEigVec = eigvecs[:,maxIndex].ravel()
                norm = np.linalg.norm(maxEigVec)

                nx[k,i,j] = maxEigVec[0]
                ny[k,i,j] = maxEigVec[1]     

                S[k,i,j] = norm
                
                
    nx = nx[zxsec,:,:]
    ny = ny[zxsec,:,:]
    S  = np.vectorize(sqrt)(nx**2 + ny**2)
    #for defects 
    nxd = nx
    nyd = ny
    # coarse grain
    S = ndimage.generic_filter(S, np.mean, size=avg)
    nx = ndimage.generic_filter(nx, np.mean, size=avg)
    ny = ndimage.generic_filter(ny, np.mean, size=avg)
    (LX, LY) = S.shape
    
    
    ## construct nematic lines
    x = []
    y = []
    for i, j in product(np.arange(LX, step=avg),
                        np.arange(LY, step=avg)):
        f = avg*(S[i, j] if scale else 1.)
        x.append(i + .5 - f*nx[i, j]/2.)
        x.append(i + .5 + f*nx[i, j]/2.)
        x.append(None)
        y.append(j + .5 - f*ny[i, j]/2.)
        y.append(j + .5 + f*ny[i, j]/2.)
        y.append(None)

        
    _update_dict(kwargs, 'linestyle', '-')
    _update_dict(kwargs, 'linewidth', 0.15)
    engine.plot(y, x,color='r', **kwargs)
    #return nxd,nyd
    return nx,ny
'''
    
def director_fields_for_defects(Q00, Q01, Q02, Q11, Q12, Q22, zxsec, avg=1, scale=False, engine=plt, **kwargs):
    """
    Plot director field associated with a given nematic field.

    Args:
        Qxx, Qxy: Components of the nematic field.
        avg: Coarse-graining size.
        scale: Scale factor that controls the size of the director.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the plot function.
    """

    (LZ, LX, LY) = Q00.shape

    nx = np.zeros((LZ,LX,LY))
    ny = np.zeros((LZ,LX,LY))
    S  = np.zeros((LZ,LX,LY))
    
    for k in range(LZ):
        for i in range(LX):
            for j in range(LY):
                q00 = Q00[k,i,j]
                q01 = Q01[k,i,j]
                q02 = Q02[k,i,j]
                q11 = Q11[k,i,j]
                q12 = Q12[k,i,j]
                q22 = Q22[k,i,j]
                A = np.array([[q00,q01,q02],[q01,q11,q12],[q02,q12,q22]])

                eigvals, eigvecs = LA.eig(A)
                idx = eigvals.argsort()[::-1]   
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:,idx]
                maxIndex = np.where(eigvals == np.amax(eigvals))
                maxEigVec = eigvecs[:,maxIndex].ravel()
                norm = np.linalg.norm(maxEigVec)

                nx[k,i,j] = maxEigVec[0]
                ny[k,i,j] = maxEigVec[1]     

                S[k,i,j] = norm
                
                
    nx = nx[zxsec,:,:]
    ny = ny[zxsec,:,:]
    S  = np.vectorize(sqrt)(nx**2 + ny**2)
    
    #for defects 
    nxd = nx
    nyd = ny
    
    return nxd,nyd

def get_director_fields(Q00, Q01, Q02, Q11, Q12, Q22, avg=1):
    """
    Plot director field associated with a given nematic field.

    Args:
        Qxx, Qxy: Components of the nematic field.
        avg: Coarse-graining size.
        scale: Scale factor that controls the size of the director.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the plot function.
    """

    (LZ, LX, LY) = Q00.shape

    nx = np.zeros((LZ,LX,LY))
    ny = np.zeros((LZ,LX,LY))
    nz = np.zeros((LZ,LX,LY))
    
    for k in range(LZ):
        for i in range(LX):
            for j in range(LY):
                q00 = Q00[k,i,j]
                q01 = Q01[k,i,j]
                q02 = Q02[k,i,j]
                q11 = Q11[k,i,j]
                q12 = Q12[k,i,j]
                q22 = Q22[k,i,j]
                A = np.array([[q00,q01,q02],[q01,q11,q12],[q02,q12,q22]])

                eigvals, eigvecs = LA.eig(A)
                idx = eigvals.argsort()[::-1]   
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:,idx]
                maxIndex = np.where(eigvals == np.amax(eigvals))
                maxEigVec = eigvecs[:,maxIndex].ravel()
                norm = np.linalg.norm(maxEigVec)

                nx[k,i,j] = maxEigVec[0]
                ny[k,i,j] = maxEigVec[1]     
                nz[k,i,j] = maxEigVec[2]     
                
    # coarse grain
    nx = ndimage.generic_filter(nx, np.mean, size=avg)
    ny = ndimage.generic_filter(ny, np.mean, size=avg)
    nz = ndimage.generic_filter(ny, np.mean, size=avg)

    return nx,ny,nz




def charge_array(nx, ny):

    """
    Compute the charge array associated with a Q-tensor field. The defects
    then show up as small regions of non-zero charge (typically 2x2).

    Args:
        Q00, Q01: The components of the nematic field.

    Returns:
        Field of the same shape as Q00 and Q01.
    """
    # compute angle
    def wang(a, b):
        """Infamous chinese function"""
        ang = atan2(abs(a[0]*b[1]-a[1]*b[0]), a[0]*b[0]+a[1]*b[1])
        if ang > pi/2.:
            b = [-i for i in b]
        m = a[0]*b[1]-a[1]*b[0]
        return -np.sign(m)*atan2(abs(m), a[0]*b[0]+a[1]*b[1])

    ## get shape and init charge array
    (LX, LY) = nx.shape
    w = np.zeros((LX, LY))

    ## we use the director field instead of Q
    S = np.vectorize(sqrt)(nx**2 + ny**2)

    # This mysterious part was stolen from Amin's code.
    for i in range(LX):
        for j in range(LY):
            ax1 = [nx[(i+1) % LX, j],
                   ny[(i+1) % LX, j]]
            ax2 = [nx[(i-1+LX) % LX, j],
                   ny[(i-1+LX) % LX, j]]
            ax3 = [nx[i, (j-1+LY) % LY],
                   ny[i, (j-1+LY) % LY]]
            ax4 = [nx[i, (j+1) % LY],
                   ny[i, (j+1) % LY]]
            ax5 = [nx[(i+1) % LX, (j-1+LY) % LY],
                   ny[(i+1) % LX, (j-1+LY) % LY]]
            ax6 = [nx[(i-1+LX) % LX, (j-1+LY) % LY],
                   ny[(i-1+LX) % LX, (j-1+LY) % LY]]
            ax7 = [nx[(i+1) % LX, (j+1) % LY],
                   ny[(i+1) % LX, (j+1) % LY]]
            ax8 = [nx[(i-1+LX) % LX, (j+1) % LY],
                   ny[(i-1+LX) % LX, (j+1) % LY]]

            w[i, j] = wang(ax1, ax5)
            w[i, j] += wang(ax5, ax3)
            w[i, j] += wang(ax3, ax6)
            w[i, j] += wang(ax6, ax2)
            w[i, j] += wang(ax2, ax8)
            w[i, j] += wang(ax8, ax4)
            w[i, j] += wang(ax4, ax7)
            w[i, j] += wang(ax7, ax1)
            w[i, j] /= 2.*pi

    return w


def defect_angle(x,y,Qxx,Qxy,w,s):

	#s = np.sign(w[i,j])    
	(LX, LY) = w.shape

	num = 0
	den = 0
	for (dx, dy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
		kk = (int(x) + LX + dx) % LX
		ll = (int(y) + LY + dy) % LY
		dxQxx = .5*(Qxx[(kk+1) % LX, ll] - Qxx[(kk-1+LX) % LX, ll])
		dxQxy = .5*(Qxy[(kk+1) % LX, ll] - Qxy[(kk-1+LX) % LX, ll])
		dyQxx = .5*(Qxx[kk, (ll+1) % LY] - Qxx[kk, (ll-1+LY) % LY])
		dyQxy = .5*(Qxy[kk, (ll+1) % LY] - Qxy[kk, (ll-1+LY) % LY])
		num += s*dxQxy - dyQxx
		den += dxQxx + s*dyQxy
	psi = .5*s/(1.-.5*s)*atan2(num, den)
	return psi

def get_defects_hybrid(w,dx,dy,Qxx,Qxy):
    
    rngA = np.arange(-1,2,1)
    rngB = np.arange(-3,4,1)
    d = []
    (LX, LY) = w.shape

    for i in range(LY):
        inew = i + 1
        for j in range(LX):
            jnew = j + 1
            yy = 0
            xx = 0
            n1 = 0
            if (abs(w[i,j]) >= 0.49 and abs(w[i,j]) < 0.51):
                                        
                ql = np.sign(w[i,j])
                yy = inew
                xx = jnew
                n1 = 1
                w[i,j] = 0
                i_n = inew 
                j_n = jnew 
                for ii in (rngA):
                    for jj in (rngA):
                        tripleti = ((i_n+ii) % (LY)) 
                        tripletj = ((j_n+jj) % (LX)) 
                        if ((ql * w[tripleti-1,tripletj-1] > 0.45) and (ql * w[tripleti-1,tripletj-1]) < 0.55 ):
                            yy = yy + inew + ii 
                            xx = xx + jnew + jj 
                            n1 = n1 + 1
                            w[tripleti-1,tripletj-1] = 0
                m = 0
                i_n = inew 
                j_n = jnew
                for ii in (rngB):
                    for jj in (rngB):
                        tripleti = ((i_n+ii) % (LY)) 
                        tripletj = ((j_n+jj) % (LX)) 
                        if ((ql * w[tripleti-1,tripletj-1] > 0.45) and (ql * w[tripleti-1,tripletj-1]) < 0.55 ):
                            yy = yy + inew + ii 
                            xx = xx + jnew + jj 
                            n1 = n1 + 1
                            w[tripleti-1,tripletj-1] = 0
                            m = 1
                
                if (m == 1):
                    if (ql > 0):
                        psi = defect_angle(xx/n1,yy/n1,Qxx,Qxy,w,ql)    
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": 1,"angle": psi})
                    else: 
                        psi = defect_angle(xx/n1,yy/n1,Qxx,Qxy,w,ql)    
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": -1,"angle": psi})
                else:
                    if (ql > 0):
                        psi = defect_angle(xx/n1,yy/n1,Qxx,Qxy,w,ql)    
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": 0.5,"angle": psi})
                    else:
                        psi = defect_angle(xx/n1,yy/n1,Qxx,Qxy,w,ql)    
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": -0.5,"angle": psi})
            
    return d
    
    
    

def get_defects(w):
    
    rngA = np.arange(-1,2,1)
    rngB = np.arange(-3,4,1)
    d = []
    
    (LX, LY) = w.shape
    #w = w.T
    
    for i in range(LY):
        inew = i + 1
        for j in range(LX):
            jnew = j + 1
            yy = 0
            xx = 0
            n1 = 0
            if (abs(w[i,j]) >= 0.45 and abs(w[i,j]) < 0.55):
                ql = np.sign(w[i,j])
                yy = inew
                xx = jnew
                n1 = 1
                w[i,j] = 0
                i_n = inew - 1 
                j_n = jnew - 1
                for ii in (rngA):
                    for jj in (rngA):
                        tripleti = ((i_n+ii) % (LY))  + 1
                        tripletj = ((j_n+jj) % (LX))  + 1
                        if ((ql * w[tripleti-1,tripletj-1] > 0.45) and (ql * w[tripleti-1,tripletj-1]) < 0.55 ):
                            yy = yy + inew + ii 
                            xx = xx + jnew + jj 
                            n1 = n1 + 1
                            w[tripleti-1,tripletj-1] = 0
                m = 0
                i_n = inew - 1 
                j_n = jnew - 1
                for ii in (rngB):
                    for jj in (rngB):
                        tripleti = ((i_n+ii) % (LY)) + 1
                        tripletj = ((j_n+jj) % (LX)) + 1
                        if ((ql * w[tripleti-1,tripletj-1] > 0.45) and (ql * w[tripleti-1,tripletj-1]) < 0.55 ):
                            yy = yy + inew + ii 
                            xx = xx + jnew + jj 
                            n1 = n1 + 1
                            w[tripleti-1,tripletj-1] = 0
                            m = 1
                if (m == 1):
                    if (ql > 0):
                        d.append({"pos": np.array([xx/n1, yy/n1]),
                          "charge": 1})


                    else: 
                        d.append({"pos": np.array([xx/n1, yy/n1]),
                          "charge": -1})


                else:
                    if (ql > 0):
                        d.append({"pos": np.array([xx/n1, yy/n1]),
                          "charge": 0.5})


                    else:
                        d.append({"pos": np.array([xx/n1, yy/n1]),
                          "charge": -0.5})
      
    return d


def get_defects_cg(w,xcoor,ycoor):
    
    rngA = np.arange(-1,2,1)
    rngB = np.arange(-3,4,1)
    d = []    
    (LX, LY) = w.shape

    for i in range(LY):
        #inew = i + 1
        for j in range(LX):
            
            jnew = xcoor[i,j]
            inew = ycoor[i,j]
            #jnew = j + 1
            yy = 0
            xx = 0
            n1 = 0
            if (abs(w[i,j]) >= 0.45 and abs(w[i,j]) < 0.55):
                ql = np.sign(w[i,j])
                yy = inew
                xx = jnew
                n1 = 1
                w[i,j] = 0
                i_n = inew - 1 
                j_n = jnew - 1
                for ii in (rngA):
                    for jj in (rngA):
                        tripleti = ((i_n+ii) % (LY))  + 1
                        tripletj = ((j_n+jj) % (LX))  + 1
                        if ((ql * w[tripleti-1,tripletj-1] > 0.45) and (ql * w[tripleti-1,tripletj-1]) < 0.55 ):
                            yy = yy + inew + ii 
                            xx = xx + jnew + jj 
                            n1 = n1 + 1
                            w[tripleti-1,tripletj-1] = 0
                m = 0
                i_n = inew - 1 
                j_n = jnew - 1
                for ii in (rngB):
                    for jj in (rngB):
                        tripleti = ((i_n+ii) % (LY)) + 1
                        tripletj = ((j_n+jj) % (LX)) + 1
                        if ((ql * w[tripleti-1,tripletj-1] > 0.45) and (ql * w[tripleti-1,tripletj-1]) < 0.55 ):
                            yy = yy + inew + ii 
                            xx = xx + jnew + jj 
                            n1 = n1 + 1
                            w[tripleti-1,tripletj-1] = 0
                            m = 1
                if (m == 1):
                    if (ql > 0):
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": 1})
                    else: 
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": -1})
                else:
                    if (ql > 0):
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": 0.5})
                    else:
                        d.append({"pos": np.array([xx/n1, yy/n1]),"charge": -0.5})
    return d


def get_defects_2D(w, Qxx, Qxy):
    """
    Returns list of defects from charge array.

    Args:
        w: Charge array.

    Returns:
        List of the form [ [ (x, y), charge] ].
    """
    # defects show up as 2x2 regions in the charge array w and must be
    # collapsed to a single point by taking the average position of
    # neighbouring points with the same charge (by breath first search).

    # bfs recursive function
    def collapse(i, j, s, x=0, y=0, n=0):
        if s*w[i, j] > .4:
            x += i + 0.
            y += j + 0.
            n += 1
            w[i, j] = 0
            collapse((i+1) % LX, j, s, x, y, n)
            collapse((i-1+LX) % LX, j, s, x, y, n)
            collapse(i, (j+1) % LY, s, x, y, n)
            collapse(i, (j-1+LY) % LY, s, x, y, n)
        return x/n, y/n

    (LX, LY) = w.shape
    d = []
    
    Qxx = Qxx.T
    Qxy = Qxy.T

    for i in range(LX):
        for j in range(LY):
            #if abs(w[i, j]) > 0.4:
            if (abs(w[i,j]) >= 0.49 and abs(w[i,j]) < 0.51):
                # charge sign
                s = np.sign(w[i, j])
                # bfs
                x, y = collapse(i, j, s)
                # compute angle, see doi:10.1039/c6sm01146b
                num = 0
                den = 0
                for (dx, dy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
                    # coordinates of nodes around the defect
                    kk = (int(x) + LX + dx) % LX
                    ll = (int(y) + LY + dy) % LY
                    # derivative at these points
                    dxQxx = .5*(Qxx[(kk+1) % LX, ll] - Qxx[(kk-1+LX) % LX, ll])
                    dxQxy = .5*(Qxy[(kk+1) % LX, ll] - Qxy[(kk-1+LX) % LX, ll])
                    dyQxx = .5*(Qxx[kk, (ll+1) % LY] - Qxx[kk, (ll-1+LY) % LY])
                    dyQxy = .5*(Qxy[kk, (ll+1) % LY] - Qxy[kk, (ll-1+LY) % LY])
                    # accumulate numerator and denominator
                    num += s*dxQxy - dyQxx
                    den += dxQxx + s*dyQxy
                #psi = s/(2.-s)*atan2(num, den)
                psi = .5*s/(1.-.5*s)*atan2(num, den)
                # add defect to list
                d.append({"pos": np.array([x, y]),
                          "charge": .5*s,
                          "angle": psi})
    return d

def get_defects_for_stress(w,Qxx,Qxy,xcoor,ycoor,avg):


    # bfs recursive function
    def collapse(i, j, s, x=0, y=0, n=0):
        if s*w[i, j] > .4:
            #x += i + 0.
            x += ycoor[i,j] + avg/2.
            #y += j + 0.
            y += xcoor[i,j] + avg/2.
            n += 1
            w[i, j] = 0
            collapse((i+1) % LX, j, s, x, y, n)
            collapse((i-1+LX) % LX, j, s, x, y, n)
            collapse(i, (j+1) % LY, s, x, y, n)
            collapse(i, (j-1+LY) % LY, s, x, y, n)
        return x/n, y/n

    (LX, LY) = w.shape
    d = []
    
    #Qxx = Qxx.T
    #Qxy = Qxy.T
    
    for i in range(LX):
        for j in range(LY):
            #if abs(w[i, j]) > 0.4:
            if (abs(w[i,j]) >= 0.45 and abs(w[i,j]) < 0.55):
                # charge sign
                s = np.sign(w[i, j])
                # bfs
                x, y = collapse(i, j, s)
                # compute angle, see doi:10.1039/c6sm01146b
                num = 0
                den = 0
                for (dx, dy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
                    # coordinates of nodes around the defect
                    kk = (int(x) + LX + dx) % LX
                    ll = (int(y) + LY + dy) % LY
                    # derivative at these points
                    dxQxx = .5*(Qxx[(kk+1) % LX, ll] - Qxx[(kk-1+LX) % LX, ll])
                    dxQxy = .5*(Qxy[(kk+1) % LX, ll] - Qxy[(kk-1+LX) % LX, ll])
                    dyQxx = .5*(Qxx[kk, (ll+1) % LY] - Qxx[kk, (ll-1+LY) % LY])
                    dyQxy = .5*(Qxy[kk, (ll+1) % LY] - Qxy[kk, (ll-1+LY) % LY])
                    # accumulate numerator and denominator
                    num += s*dxQxy - dyQxx
                    den += dxQxx + s*dyQxy
                #psi = s/(2.-s)*atan2(num, den)
                psi = .5*s/(1.-.5*s)*atan2(num, den)
                # add defect to list
                d.append({"pos": np.array([x, y]),
                          "charge": .5*s,
                          "angle": psi})
    return d
    
    
    
            
def defects_2D(dx,dy,Qxx,Qxy, engine=plt, orient=False):

    w = charge_array(dx,dy)
    #w = -w
    w = w.T
    defects = get_defects_2D(w,Qxx,Qxy)
    Ltri = 0.08
    
    for d in defects:
        angle = d['angle']
        pt1 = np.array([Ltri * cos(angle), Ltri * sin(angle)])
        if d['charge'] == 0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='b')
        elif d['charge'] == -0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='g')
        elif d['charge'] == 1.:
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='k')
        elif d['charge'] == -1.:
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='y')
            
            

    
        
def defects(dx,dy, engine=plt, arrow_len=8):

    w = charge_array(dx,dy)
    defects = get_defects(w)
    for d in defects:
        if d['charge'] == 0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=2)
        elif d['charge'] == -0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=2)

        elif d['charge'] == 1.:
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=2)

        elif d['charge'] == -1.:
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=2)


def defects_hybrid(dx,dy,Qxx,Qxy,half_int=True,full_int=False,orient=True, engine=plt):

    w = charge_array(dx,dy)    
    #w = -w
    #Qxx = Qxx.T
    #Qxy = Qxy.T
    #w = w.T
    defects = get_defects_hybrid(w,dx,dy,Qxx,Qxy)
    Ltri = 0.08

    for d in defects:
        angle = d['angle']
        pt1 = np.array([Ltri * cos(angle), Ltri * sin(angle)])
        if (d['charge'] == 0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='b')
        elif (d['charge'] == -0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='g')
        elif (d['charge'] == 1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='k')
        elif (d['charge'] == -1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='y')
            
            
def defects_hybrid_cg(dx,dy,avg=1,half_int=True,full_int=False,orient=True, engine=plt):


    (LX, LY) = dx.shape
    LX_cg = len(np.arange(LX,step=avg))
    LY_cg = len(np.arange(LY,step=avg))
    nx_cg = np.zeros((LX_cg,LY_cg))
    ny_cg = np.zeros((LX_cg,LY_cg))
    xcoor = np.zeros((LX_cg,LY_cg))
    ycoor = np.zeros((LX_cg,LY_cg))
    for i, j in product(np.arange(LX, step=avg),np.arange(LY, step=avg)):
    	nx_cg[int(i/avg),int(j/avg)] = dx[i,j] 
    	ny_cg[int(i/avg),int(j/avg)] = dy[i,j]
    	xcoor[int(i/avg),int(j/avg)] = i
    	ycoor[int(i/avg),int(j/avg)] = j
    	
    w = charge_array(nx_cg,ny_cg)    
    defects = get_defects_cg(w,xcoor,ycoor)
    Ltri = 0.08

    for d in defects:
        angle = d['angle']
        pt1 = np.array([Ltri * cos(angle), Ltri * sin(angle)])
        if (d['charge'] == 0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='b')
        elif (d['charge'] == -0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='g')
        elif (d['charge'] == 1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='k')
        elif (d['charge'] == -1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='y')      
            	

def defects_hybrid_stats(dx,dy,Qxx,Qxy):

    w = charge_array(dx,dy)    
    #w = -w
    #Qxx = Qxx.T
    #Qxy = Qxy.T
    #w = w.T
    defects = get_defects_hybrid(w,dx,dy,Qxx,Qxy)
    return defects 

            	
            	
            	        	   

def shape_field_2D(frame, zxsec, size=1, avg=1, show_def=False, arrow_len=0,
                engine=plt, **kwargs):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        avg: Average size (reduces the number of points plotted)
        show_def: If true, show defects.
        arrow_len: If non-zero, prints defect speed.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the director function.
    """

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz) = get_nematic_field(frame.phi, frame.S00, frame.S01, frame.S02, frame.S11, frame.S12, frame.S22,
                                   size=1, mode=mode)
    Qxx *= (1.-frame.parameters['walls'])
    Qxy *= (1.-frame.parameters['walls'])
    Qxz *= (1.-frame.parameters['walls'])
    Qyy *= (1.-frame.parameters['walls'])
    Qyz *= (1.-frame.parameters['walls'])
    Qzz *= (1.-frame.parameters['walls'])

    (nx,ny) = director(Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, zxsec, avg=avg, engine=engine, **kwargs)     
    Qxx = Qxx[zxsec,:,:]
    Qxy = Qxy[zxsec,:,:]
    Qxx = ndimage.generic_filter(Qxx, np.mean, size=avg)
    Qxy = ndimage.generic_filter(Qxy, np.mean, size=avg)

    if show_def:
    	defects_2D(nx,ny,Qxx,Qxy, engine=engine)


def get_defects_2D_cg(w,xcoor,ycoor,avg):


    # bfs recursive function
    def collapse(i, j, s, x=0, y=0, n=0):
        if s*w[i, j] > .4:
            #x += i + 0.
            x += ycoor[i,j] + avg/2.
            #y += j + 0.
            y += xcoor[i,j] + avg/2.
            n += 1
            w[i, j] = 0
            collapse((i+1) % LX, j, s, x, y, n)
            collapse((i-1+LX) % LX, j, s, x, y, n)
            collapse(i, (j+1) % LY, s, x, y, n)
            collapse(i, (j-1+LY) % LY, s, x, y, n)
        return x/n, y/n

    (LX, LY) = w.shape
    d = []
    
    for i in range(LX):
        for j in range(LY):
            #if abs(w[i, j]) > 0.4:
            if (abs(w[i,j]) >= 0.45 and abs(w[i,j]) < 0.55):
                # charge sign
                s = np.sign(w[i, j])
                # bfs
                x, y = collapse(i, j, s)
                # compute angle, see doi:10.1039/c6sm01146b
                d.append({"pos": np.array([x, y]),
                          "charge": .5*s})
    return d
    
def defects_2D_cg(dx,dy,avg=1,half_int=True,full_int=False,orient=True, engine=plt):


    (LX, LY) = dx.shape
    LX_cg = len(np.arange(LX,step=avg))
    LY_cg = len(np.arange(LY,step=avg))
    nx_cg = np.zeros((LX_cg,LY_cg))
    ny_cg = np.zeros((LX_cg,LY_cg))
    xcoor = np.zeros((LX_cg,LY_cg))
    ycoor = np.zeros((LX_cg,LY_cg))
    for i, j in product(np.arange(LX, step=avg),np.arange(LY, step=avg)):
    	nx_cg[int(i/avg),int(j/avg)] = dx[i,j] 
    	ny_cg[int(i/avg),int(j/avg)] = dy[i,j]
    	xcoor[int(i/avg),int(j/avg)] = i
    	ycoor[int(i/avg),int(j/avg)] = j
    	
    w = charge_array(nx_cg,ny_cg)    
    #w = -w
    #defects = get_defects_cg(w,xcoor,ycoor)
    defects = get_defects_2D_cg(w,xcoor,ycoor,avg)
    Ltri = 0.08

    for d in defects:
        #angle = d['angle']
        #pt1 = np.array([Ltri * cos(angle), Ltri * sin(angle)])
        if (d['charge'] == 0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=4)
            #if (orient):
            #	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='b')
        elif (d['charge'] == -0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=4)
            #if (orient):
            #	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='g')
        elif (d['charge'] == 1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=4)
            #if (orient):
            #	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='k')
        elif (d['charge'] == -1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=4)
            #if (orient):
            #	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='y')  
            
def defects_2D_for_stress(dx,dy,Qxx,Qxy,avg=1,half_int=True,full_int=False,orient=True, engine=plt):


    (LX, LY) = dx.shape
    LX_cg = len(np.arange(LX,step=avg))
    LY_cg = len(np.arange(LY,step=avg))
    nx_cg = np.zeros((LX_cg,LY_cg))
    ny_cg = np.zeros((LX_cg,LY_cg))
    xcoor = np.zeros((LX_cg,LY_cg))
    ycoor = np.zeros((LX_cg,LY_cg))
    for i, j in product(np.arange(LX, step=avg),np.arange(LY, step=avg)):
    	nx_cg[int(i/avg),int(j/avg)] = dx[i,j] 
    	ny_cg[int(i/avg),int(j/avg)] = dy[i,j]
    	xcoor[int(i/avg),int(j/avg)] = i
    	ycoor[int(i/avg),int(j/avg)] = j
    	
    w = charge_array(nx_cg,ny_cg)    
    #w = -w
    #defects = get_defects_cg(w,xcoor,ycoor)
    defects = get_defects_for_stress(w,Qxx,Qxy,xcoor,ycoor,avg)
    Ltri = 0.08

    for d in defects:
        angle = d['angle']
        pt1 = np.array([Ltri * cos(angle), Ltri * sin(angle)])
        if (d['charge'] == 0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='b')
        elif (d['charge'] == -0.5 and half_int):
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='g')
        elif (d['charge'] == 1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='k')
        elif (d['charge'] == -1. and full_int):
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=4)
            if (orient):
            	engine.quiver(d["pos"][0],d["pos"][1],pt1[0],pt1[1], scale=1,color='y')        
            
def defects_2D_cg_stats(dx,dy,avg=1):


    (LX, LY) = dx.shape
    LX_cg = len(np.arange(LX,step=avg))
    LY_cg = len(np.arange(LY,step=avg))
    nx_cg = np.zeros((LX_cg,LY_cg))
    ny_cg = np.zeros((LX_cg,LY_cg))
    xcoor = np.zeros((LX_cg,LY_cg))
    ycoor = np.zeros((LX_cg,LY_cg))
    for i, j in product(np.arange(LX, step=avg),np.arange(LY, step=avg)):
    	nx_cg[int(i/avg),int(j/avg)] = dx[i,j] 
    	ny_cg[int(i/avg),int(j/avg)] = dy[i,j]
    	xcoor[int(i/avg),int(j/avg)] = i
    	ycoor[int(i/avg),int(j/avg)] = j
    	
    w = charge_array(nx_cg,ny_cg)    
    defects = get_defects_2D_cg(w,xcoor,ycoor)
    return defects 
    
    
            
def shape_field_2D_cg(frame, zxsec, size=1, avg=1, show_def=False, half_int=True,full_int=False,orient=False,
                engine=plt, **kwargs):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        avg: Average size (reduces the number of points plotted)
        show_def: If true, show defects.
        arrow_len: If non-zero, prints defect speed.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the director function.
    """

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz) = get_nematic_field(frame.phi, frame.S00, frame.S01, frame.S02, frame.S11, frame.S12, frame.S22,
                                   size=1, mode=mode)
    Qxx *= (1.-frame.parameters['walls'])
    Qxy *= (1.-frame.parameters['walls'])
    Qxz *= (1.-frame.parameters['walls'])
    Qyy *= (1.-frame.parameters['walls'])
    Qyz *= (1.-frame.parameters['walls'])
    Qzz *= (1.-frame.parameters['walls'])

    (nx,ny) = director(Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, zxsec, avg=avg, engine=engine, **kwargs)     
    Qxx = Qxx[zxsec,:,:]
    Qxy = Qxy[zxsec,:,:]
    Qxx = ndimage.generic_filter(Qxx, np.mean, size=avg)
    Qxy = ndimage.generic_filter(Qxy, np.mean, size=avg)

    if show_def:
    	defects_2D_cg(nx,ny,avg,half_int,full_int,orient, engine=engine) 	
    	

    	
def shape_field_hybrid(frame, zxsec, size=1, avg=1, show_def=False, half_int=True,full_int=False,orient=True,
                engine=plt, **kwargs):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        avg: Average size (reduces the number of points plotted)
        show_def: If true, show defects.
        arrow_len: If non-zero, prints defect speed.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the director function.
    """

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz) = get_nematic_field(frame.phi, frame.S00, frame.S01, frame.S02, frame.S11, frame.S12, frame.S22,
                                   size=1, mode=mode)
    Qxx *= (1.-frame.parameters['walls'])
    Qxy *= (1.-frame.parameters['walls'])
    Qxz *= (1.-frame.parameters['walls'])
    Qyy *= (1.-frame.parameters['walls'])
    Qyz *= (1.-frame.parameters['walls'])
    Qzz *= (1.-frame.parameters['walls'])

    (nx,ny) = director(Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, zxsec, avg=avg, engine=engine, **kwargs)     
    Qxx = Qxx[zxsec,:,:]
    Qxy = Qxy[zxsec,:,:]
    Qxx = ndimage.generic_filter(Qxx, np.mean, size=avg)
    Qxy = ndimage.generic_filter(Qxy, np.mean, size=avg)

    if show_def:
    	defects_hybrid(nx,ny,Qxx,Qxy,half_int,full_int,orient, engine=engine) 	         



def shape_field(frame, zxsec, size=1, avg=1, show_def=False, arrow_len=0,
                engine=plt, **kwargs):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        avg: Average size (reduces the number of points plotted)
        show_def: If true, show defects.
        arrow_len: If non-zero, prints defect speed.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the director function.
    """

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz) = get_nematic_field(frame.phi, frame.S00, frame.S01, frame.S02, frame.S11, frame.S12, frame.S22,
                                   size=size, mode=mode)
    Qxx *= (1.-frame.parameters['walls'])
    Qxy *= (1.-frame.parameters['walls'])
    Qxz *= (1.-frame.parameters['walls'])
    Qyy *= (1.-frame.parameters['walls'])
    Qyz *= (1.-frame.parameters['walls'])
    Qzz *= (1.-frame.parameters['walls'])

    (nx,ny) = director(Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, zxsec, avg=avg, engine=engine, **kwargs)     

    if show_def:
    	defects(nx,ny, engine=engine)
    	
    	
def shape_field_defects(frame, zxsec, size=1, avg=1,engine=plt, **kwargs):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        avg: Average size (reduces the number of points plotted)
        show_def: If true, show defects.
        arrow_len: If non-zero, prints defect speed.
        engine: Plotting engine or axis.
        **kwargs: Keyword arguments passed to the director function.
    """

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz) = get_nematic_field(frame.phi, frame.S00, frame.S01, frame.S02, frame.S11, frame.S12, frame.S22,
                                   size=size, mode=mode)
    Qxx *= (1.-frame.parameters['walls'])
    Qxy *= (1.-frame.parameters['walls'])
    Qxz *= (1.-frame.parameters['walls'])
    Qyy *= (1.-frame.parameters['walls'])
    Qyz *= (1.-frame.parameters['walls'])
    Qzz *= (1.-frame.parameters['walls'])

    (nx,ny) = director_fields_for_defects(Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, zxsec, avg=avg, engine=engine, **kwargs)     
    defects(nx,ny, engine=engine)

   

def get_shape_directors(frame, size=1, avg=1):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    (Qxx, Qxy, Qxz, Qyy, Qyz, Qzz) = get_nematic_field(frame.phi, frame.S00, frame.S01, frame.S02, frame.S11, frame.S12, frame.S22,
                                   size=size, mode=mode)
    Qxx *= (1.-frame.parameters['walls'])
    Qxy *= (1.-frame.parameters['walls'])
    Qxz *= (1.-frame.parameters['walls'])
    Qyy *= (1.-frame.parameters['walls'])
    Qyz *= (1.-frame.parameters['walls'])
    Qzz *= (1.-frame.parameters['walls'])

    (nx,ny,nz) = get_director_fields(Qxx, Qxy, Qxz, Qyy, Qyz, Qzz, avg=avg)     
    return nx,ny,nz




def stress_tensor_field(fig,frame, zxsec=0,size=1,cgL=2, engine=plt,plotSig = 1, cbar=False):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy, fz = get_force_field(frame.phi, frame.parameters['xi']*frame.velocity-frame.Fpol, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls']) 
    (LZ, LX, LY) = fx.shape

    cut_fx = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fx)
    cut_fy = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fy)
    cut_fz = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fz)
    (LZ, LX, LY) = cut_fx.shape

    resized_fx = resize_data_for_cg(cgL,LZ,LY,LX,cut_fx)
    resized_fy = resize_data_for_cg(cgL,LZ,LY,LX,cut_fy)
    resized_fz = resize_data_for_cg(cgL,LZ,LY,LX,cut_fz)
    (LZn, LXn, LYn) = resized_fx.shape

    sigXX,sigYY,sigZZ,sigXY,sigXZ,sigYZ = get_stress_Tensor(cgL,LZn,LYn,LXn,resized_fx,resized_fy,resized_fz)

    if (plotSig == 1):
    	sig = sigXX[zxsec,:,:]
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)
    	caxx = engine.contourf(x, y, sig, 20,cmap='RdBu', origin = 'lower')
    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	cmap = mpl.cm.RdBu
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    	
    	
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	if(cbar):
    		cbar_ax = fig.add_axes([0.3, 0.075, 0.4, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		cbar.set_label('$\sigma_{xx}$', labelpad=0.0)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()
    		
    		
    if (plotSig == 2):
    	sig = sigYY[zxsec,:,:]
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)
    	caxx = engine.contourf(x, y, sig, 20,cmap='RdBu', origin = 'lower')
    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	cmap = mpl.cm.RdBu
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    		
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	if(cbar):
    		cbar_ax = fig.add_axes([0.3, 0.075, 0.4, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		cbar.set_label('$\sigma_{yy}$', labelpad=0.0)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()
    		
    		
    if (plotSig == 3):
    	sig = sigZZ[zxsec,:,:]
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)

    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	cmap = mpl.cm.RdBu
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    		
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	if(cbar):
    		cbar_ax = fig.add_axes([0.3, 0.075, 0.4, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		cbar.set_label('$\sigma_{zz}$', labelpad=0.0)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()


    		
    		
    		
    if (plotSig == 4):
    	sig = sigXY[zxsec,:,:]
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)
    	caxx = engine.contourf(x, y, sig, 20,cmap='RdBu', origin = 'lower')
    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	cmap = mpl.cm.RdBu
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    		
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	if(cbar):
    		cbar_ax = fig.add_axes([0.3, 0.075, 0.4, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		cbar.set_label('$\sigma_{xy}$', labelpad=0.0)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()
    		
    if (plotSig == 5):
    	sig = sigXZ[zxsec,:,:]
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)
    	minVal = round_sig(minVal, sig=2)
    	maxVal = round_sig(maxVal, sig=2)
    	#minVal = -0.001
    	#maxVal = 0.001
    	caxx = engine.contourf(x, y, sig, 20,cmap='RdBu', origin = 'lower')
    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	cmap = mpl.cm.RdBu
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0.,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    		
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	if(cbar):
    		cbar_ax = fig.add_axes([0.4, -0.005, 0.2, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0.,maxVal],cax=cbar_ax)
    		#cbar.set_label('pressure, $p$', labelpad=0.0, size='large')
    		cbar.ax.set_title('shear stress, $\sigma_{xz}$', fontsize=20)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()
    		
    if (plotSig == 6):
    	sig = sigYZ[zxsec,:,:]
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)
    	caxx = engine.contourf(x, y, sig, 20,cmap='RdBu', origin = 'lower')
    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    		
    	cmap = mpl.cm.RdBu
    	normi = mpl.colors.DivergingNorm(vmin=minVal,
                             vcenter=0,
                             vmax=maxVal)
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	if(cbar):
    		cbar_ax = fig.add_axes([0.3, 0.075, 0.4, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		cbar.set_label('$\sigma_{yz}$', labelpad=0.0)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()
    		
    if (plotSig == 7):
    	sig = (1./3.)*np.array( ( sigXX[zxsec,:,:] + sigYY[zxsec,:,:] + sigZZ[zxsec,:,:] ))
    	(nx,ny) = sig.shape
    	x = np.linspace(0, LX-1, nx)
    	y = np.linspace(0, LY-1, ny)
    	minVal = np.min(sig)
    	maxVal = np.max(sig)
    	#minVal = -0.0021
    	#maxVal = +0.0021
    	minVal = round_sig(minVal, sig=2)
    	maxVal = round_sig(maxVal, sig=2)
    	#minVal = -0.0021
    	#maxVal = 0.0021
    	caxx = engine.contourf(x, y, sig, 20,cmap='RdBu', origin = 'lower')
    	#print('min: ',np.min(sig))
    	#print('max: ',np.max(sig))
    	cmap = mpl.cm.RdBu
    	if (minVal < 0 and maxVal > 0):
    		normi = mpl.colors.DivergingNorm(vmin=minVal,vcenter=0,vmax=maxVal)
    	else:
    		normi = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    		
    	caxx = engine.contourf(x, y, sig, 20,norm=normi,cmap=cmap, origin = 'lower')
    	'''
    	if(cbar):
    		cbar_ax = fig.add_axes([0.17, 0.3, 0.02, 0.4]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="vertical", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		cbar.set_label('$pressure, p$', labelpad=-55.0, size='large')
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.ax.yaxis.set_ticks_position('left')
    		cbar.ax.yaxis.set_offset_position('right')
    		cbar.update_ticks()
    	'''
    	if(cbar):
    		cbar_ax = fig.add_axes([0.4, -0.005, 0.2, 0.02]) #left bottom width height 
    		cbar = plt.colorbar(caxx,orientation="horizontal", ticks=[minVal,0,maxVal],cax=cbar_ax)
    		#cbar.set_label('pressure, $p$', labelpad=0.0, size='large')
    		cbar.ax.set_title('isotropic stress, $\sigma^{Iso}$', fontsize=20)
    		cbar.formatter.set_powerlimits((0, 0))
    		cbar.update_ticks()
    	


def pdf_stress(fname,f1, cgL,size, engine=plt, num_bins =100):


    mode = 'wrap' if f1.parameters['BC'] == 0 else 'constant'
    f1x, f1y, f1z = get_force_field(f1.phi, f1.parameters['xi']*f1.velocity-f1.Fpol, size, mode=mode)
    f1x *= (1.-f1.parameters['walls'])
    f1y *= (1.-f1.parameters['walls'])
    f1z *= (1.-f1.parameters['walls']) 
    (LZ, LX, LY) = f1x.shape

    cut_f1x = cut_wall_for_cg(LZ,LY,LX,f1.parameters['wall_thickness'],f1x)
    cut_f1y = cut_wall_for_cg(LZ,LY,LX,f1.parameters['wall_thickness'],f1y)
    cut_f1z = cut_wall_for_cg(LZ,LY,LX,f1.parameters['wall_thickness'],f1z)
    (LZ, LX, LY) = cut_f1x.shape

    resized_f1x = resize_data_for_cg(cgL,LZ,LY,LX,cut_f1x)
    resized_f1y = resize_data_for_cg(cgL,LZ,LY,LX,cut_f1y)
    resized_f1z = resize_data_for_cg(cgL,LZ,LY,LX,cut_f1z)
    (LZn, LXn, LYn) = resized_f1x.shape

    sig1XX,sig1YY,sig1ZZ,sig1XY,sig1XZ,sig1YZ = get_stress_Tensor(cgL,LZn,LYn,LXn,resized_f1x,resized_f1y,resized_f1z)
    
    s1xxs = get_solid_stress(f1,sig1XX)
    s1yys = get_solid_stress(f1,sig1YY)
    s1zzs = get_solid_stress(f1,sig1ZZ)
    s1xzs = get_solid_stress(f1,sig1XZ)
    p1 = (1./3.)*np.array ( (s1xxs+s1yys+s1zzs) )
    ffile = fname + '.dat'
    np.savetxt(ffile,s1xzs)
    #np.savetxt(ffile,p1)
    
    
    
#    mode = 'wrap' if f2.parameters['BC'] == 0 else 'constant'
#    f2x, f2y, f2z = get_force_field(f2.phi, f2.parameters['xi']*f2.velocity-f2.Fpol, size, mode=mode)
#    f2x *= (1.-f2.parameters['walls'])
#    f2y *= (1.-f2.parameters['walls'])
#    f2z *= (1.-f2.parameters['walls']) 
#    (LZ, LX, LY) = f2x.shape

#    cut_f2x = cut_wall_for_cg(LZ,LY,LX,f2.parameters['wall_thickness'],f2x)
#    cut_f2y = cut_wall_for_cg(LZ,LY,LX,f2.parameters['wall_thickness'],f2y)
#    cut_f2z = cut_wall_for_cg(LZ,LY,LX,f2.parameters['wall_thickness'],f2z)
#    (LZ, LX, LY) = cut_f2x.shape

#    resized_f2x = resize_data_for_cg(cgL,LZ,LY,LX,cut_f2x)
#    resized_f2y = resize_data_for_cg(cgL,LZ,LY,LX,cut_f2y)
#    resized_f2z = resize_data_for_cg(cgL,LZ,LY,LX,cut_f2z)
#    (LZn, LXn, LYn) = resized_f2x.shape
#
#    sig2XX,sig2YY,sig2ZZ,sig2XY,sig2XZ,sig2YZ = get_stress_Tensor(cgL,LZn,LYn,LXn,resized_f2x,resized_f2y,resized_f2z)
#    
#    s2xxs = get_solid_stress(f2,sig2XX)
#    s2yys = get_solid_stress(f2,sig2YY)
#    s2zzs = get_solid_stress(f2,sig2ZZ)
#    p2 = (1./3.)*(s2xxs+s2yys+s2zzs)
    
    n, bins, patches = engine.hist(p1, num_bins,  
                            density = True,  
                            color ='black', 
                            alpha = 0.7, label = 'p1')
    engine.legend(frameon=False)
                  
#    n, bins, patches = engine.hist(p2, num_bins,  
#                            density = True,  
#                            color ='blue', 
#                            alpha = 0.7, label = 'p2')
#    engine.legend(frameon=False)



        
        
        
        
        
        
        
        
        
        
        
        
        
'''    
def force_field_thresholding(frame, zxsec, size=1, perC=80, dims=1, engine=plt, cbar=False):
    """
    Plot nematic field associated with the shape tensor of each cell.

    Args:
        frame: Frame to plot, from archive module.
        size: Coarse-graining size.
        engine: Plotting engine or axis.
        magn: Plot velocity magnitude as a heatmap?
        cbar: Show color bar?
        avg: Size of the averaging (drops points)
    """
    #nx = ndimage.generic_filter(nx, np.mean, size=avg)
    #ny = ndimage.generic_filter(ny, np.mean, size=avg)

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    fx, fy,fz = get_force_field(frame.phi, frame.Fpressure, size, mode=mode)
    fx *= (1.-frame.parameters['walls'])
    fy *= (1.-frame.parameters['walls'])
    fz *= (1.-frame.parameters['walls'])
    
    (LZ, LX, LY) = fx.shape
    fxs = []
    fys = []
    fzs = []
    for ii in range(len(frame.phi)):
        phi_i = frame.phi[ii]
        #for k in range(LZ):
        for i in range(LX):
        	for j in range(LY):
        		pi = phi_i[zxsec,i,j]
        		if pi > 0.5:
        			fxs.append(fx[zxsec,i,j])
        			fys.append(fy[zxsec,i,j])
        			fzs.append(fz[zxsec,i,j])
        				
    if dims == 1:
    	#fthresh = np.percentile(fxs,perC)
    	#fx[fx<fthresh] = 0
    	cax = engine.imshow(fx[zxsec,:,:], interpolation='lanczos', cmap='RdBu',origin='lower')
    	#,vmin=-0.05,vmax=0.05)
    	if cbar:
    		plt.colorbar(cax,shrink=0.3)
    if dims == 2:
    	#fthresh = np.percentile(fys,perC)
    	#fy[fy<fthresh] = 0
    	cax = engine.imshow(fy[zxsec,:,:], interpolation='lanczos', cmap='RdBu',origin='lower')
    	#,vmin=-0.05,vmax=0.05)
    	if cbar:
    		plt.colorbar(cax,shrink=0.3)
    if dims == 3:
    	#fthresh = np.percentile(fzs,perC)
    	#fz[fz<fthresh] = 0
    	cax = engine.imshow(fz[zxsec,:,:], interpolation='lanczos', cmap='RdBu',origin='lower')
    	#,vmin=-0.02,vmax=0.02)
    	if cbar:
    		plt.colorbar(cax,shrink=0.3)
'''
        
        
            
            
            
'''          
def press_field_thresholding(frame, zxsec, size, perC = 80, engine=plt, cbar=False):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    szz = frame.stress_zz
    szz = get_stress_field(szz, size, mode='wrap')
    szz *= (1.-frame.parameters['walls'])
    
    (LZ, LX, LY) = szz.shape
    szzs = []
    for ii in range(len(frame.phi)):
        phi_i = frame.phi[ii]
        #for k in range(LZ):
        for i in range(LX):
        	for j in range(LY):
        		pi = phi_i[zxsec,i,j]
        		if pi > 0.5:
        			szzs.append(szz[zxsec,i,j])

    sthresh = np.percentile(szzs, perC)
    szz[szz < sthresh] = 0
    
    cax = engine.imshow(szz[zxsec,:,:], interpolation='lanczos', cmap='Reds',
                            origin='lower')
    if cbar:
    	plt.colorbar(cax,shrink=0.3)
'''

'''         
def press_field_flucs(frame, zxsec, size, minVal,maxVal,engine=plt, cbar=False):

    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
    sxx = frame.stress_xx
    sxx = get_stress_field(sxx, size, mode='wrap')
    sxx *= (1.-frame.parameters['walls'])
    (LZ,LX,LY) = sxx.shape
    sigs = []
    for ii in range(len(frame.phi)):
        phi_i = frame.phi[ii]
        for k in range(LZ):
        	for i in range(LX):
        		for j in range(LY):
        			pi = phi_i[k,i,j]
        			if pi > 0.5:
        				sigs.append(sxx[k,i,j])
        				
    sxx = sxx[zxsec,:,:] - np.mean( sigs )				
    cax = engine.imshow(sxx, interpolation='lanczos', cmap='RdBu', origin = 'lower')
    #,vmin=minVal,vmax=maxVal)
    print('min = ',np.min(sxx))
    print('max = ',np.max(sxx))
    plt.colorbar(cax)
    #,vmin=-0.0007,vmax=0.0002)
    #if cbar:
    #	plt.colorbar(cax)  
''' 
            
'''
def diff_press_field(ft1,ft2, zxsec, size, minVal,maxVal,engine=plt, cbar=False):

    mode = 'wrap' if ft1.parameters['BC'] == 0 else 'constant'
    st1xx = ft1.stress_xx
    st2xx = ft2.stress_xx
    st1xx = get_stress_field(st1xx, size, mode='wrap')
    st1xx *= (1.-ft1.parameters['walls'])
    st2xx = get_stress_field(st2xx, size, mode='wrap')
    st2xx *= (1.-ft1.parameters['walls'])
    sdif = st1xx - st2xx
    #(LZ,LX,LY) = sdif.shape
    #sigs = []
    #for ii in range(len(frame.phi)):
    #    phi_i = frame.phi[ii]
    #    for k in range(LZ):
    #    	for i in range(LX):
    #    		for j in range(LY):
    #    			pi = phi_i[k,i,j]
    #    			if pi > 0.5:
    #    				sigs.append(sdif[k,i,j])
    
    
    sdif = sdif[zxsec,:,:] - np.mean( sdif[zxsec,:,:].ravel() )
    
    cax = engine.imshow(sdif, interpolation='lanczos', cmap='RdBu', origin = 'lower')
    #,vmin=minVal,vmax=maxVal)
    print('min = ',np.min(sdif))
    print('max = ',np.max(sdif))
    #,vmin=-0.0007,vmax=0.0002)
    if cbar:
    	plt.colorbar(cax)
    #return np.mean(sigs),np.var(sigs),skewness(sigs),kurtosis(sigs)
'''           
            
'''
#def stress_tensor_field_thresholding(frame, zxsec=0,size=1,cgL=2, engine=plt, cbar=False):
#
#    mode = 'wrap' if frame.parameters['BC'] == 0 else 'constant'
#    fx, fy, fz = get_force_field(frame.phi, frame.parameters['xi']*frame.velocity-frame.Fpol, size, mode=mode)
#    fx *= (1.-frame.parameters['walls'])
#    fy *= (1.-frame.parameters['walls'])
#    fz *= (1.-frame.parameters['walls']) 
#    (LZ, LX, LY) = fx.shape
#    #print('shape before',fx.shape)
#    cut_fx = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fx)
#    cut_fy = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fy)
#    cut_fz = cut_wall_for_cg(LZ,LY,LX,frame.parameters['wall_thickness'],fz)
#    (LZ, LX, LY) = cut_fx.shape
#    #print('shape after wall remvoal', cut_fx.shape)
#    resized_fx = resize_data_for_cg(cgL,LZ,LY,LX,cut_fx)
#    resized_fy = resize_data_for_cg(cgL,LZ,LY,LX,cut_fy)
#    resized_fz = resize_data_for_cg(cgL,LZ,LY,LX,cut_fz)
#    (LZn, LXn, LYn) = resized_fx.shape
#    #print('shape after resize', resized_fx.shape)
#    #nx = ndimage.generic_filter(nx, np.mean, size=avg)
#    sigXX,sigXY,sigXZ,sigYX,sigYY,sigYZ,sigZX,sigZY,sigZZ = get_stress_Tensor(cgL,LZn,LYn,LXn,resized_fx,resized_fy,resized_fz)
#    #sxx *= (1.-frame.parameters['walls'])
#    #mszz = np.mean(szz)
#    #mm = np.percentile(sxx, 50)
#    #sxx[sxx < mm] = 0
#    
#    cax = engine.imshow(sigXY[zxsec,:,:], interpolation='lanczos', cmap='plasma',
#                            origin='lower')
#    if cbar:
#            plt.colorbar(cax,shrink=0.3)
'''


'''
def get_defects_2D(w, Qxx, Qxy):
    """
    Returns list of defects from charge array.

    Args:
        w: Charge array.

    Returns:
        List of the form [ [ (x, y), charge] ].
    """
    # defects show up as 2x2 regions in the charge array w and must be
    # collapsed to a single point by taking the average position of
    # neighbouring points with the same charge (by breath first search).

    # bfs recursive function
    def collapse(i, j, s, x=0, y=0, n=0):
        if s*w[i, j] > .4:
            x += i + 1.5
            y += j + 1.5
            n += 1
            w[i, j] = 0
            collapse((i+1) % LX, j, s, x, y, n)
            collapse((i-1+LX) % LX, j, s, x, y, n)
            collapse(i, (j+1) % LY, s, x, y, n)
            collapse(i, (j-1+LY) % LY, s, x, y, n)
        return x/n, y/n

    (LX, LY) = w.shape
    d = []
    
    #Qxx = Qxx.T
    #Qxy = Qxy.T

    for i in range(LX):
        for j in range(LY):
            #if abs(w[i, j]) > 0.4:
            if (abs(w[i,j]) >= 0.45 and abs(w[i,j]) < 0.55):
                # charge sign
                s = np.sign(w[i, j])
                # bfs
                x, y = collapse(i, j, s)
                # compute angle, see doi:10.1039/c6sm01146b
                num = 0
                den = 0
                for (dx, dy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
                    # coordinates of nodes around the defect
                    kk = (int(x) + LX + dx) % LX
                    ll = (int(y) + LY + dy) % LY
                    # derivative at these points
                    dxQxx = .5*(Qxx[(kk+1) % LX, ll] - Qxx[(kk-1+LX) % LX, ll])
                    dxQxy = .5*(Qxy[(kk+1) % LX, ll] - Qxy[(kk-1+LX) % LX, ll])
                    dyQxx = .5*(Qxx[kk, (ll+1) % LY] - Qxx[kk, (ll-1+LY) % LY])
                    dyQxy = .5*(Qxy[kk, (ll+1) % LY] - Qxy[kk, (ll-1+LY) % LY])
                    # accumulate numerator and denominator
                    num += s*dxQxy - dyQxx
                    den += dxQxx + s*dyQxy
                #psi = s/(2.-s)*atan2(num, den)
                psi = .5*s/(1.-.5*s)*atan2(num, den)
                # add defect to list
                d.append({"pos": np.array([x, y]),
                          "charge": .5*s,
                          "angle": psi})
    return d
'''
    
'''     
def defects_2D(dx,dy,Qxx,Qxy, engine=plt, arrow_len=8):

    w = charge_array(dx,dy)
    w = w.T
    defects = get_defects_2D(w,Qxx,Qxy)
    
    for d in defects:
        if d['charge'] == 0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'b<', markersize=2)
        elif d['charge'] == -0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'g>', markersize=2)

        elif d['charge'] == 1.:
            engine.plot(d["pos"][0], d["pos"][1], 'k*', markersize=2)

        elif d['charge'] == -1.:
            engine.plot(d["pos"][0], d["pos"][1], 'ys', markersize=2)
'''
        
        
        
        
        
        


