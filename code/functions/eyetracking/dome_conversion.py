import numpy as np
import sys
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/RF_VR_mapping/eyetracking')
import time_conversion as tc
import irec_conversion
from irec_conversion import dome2cartesian, normalize, angle2direction

# new dome coordinates to eye coordinates

# X = Right of center is positive
# Y = Above center is positive
# Z = In front of center is positive

def my_dot(a,b,axis=1):
    return np.sum(a*b,axis=axis)

def calc_irec_rotation(irec_x,irec_y):
    eye_pos = angle2direction(irec_x, irec_y)
    eye_center = np.broadcast_to(sphere2cartesian(0,0,1), eye_pos.shape)
    rot_axes = normalize(np.cross(eye_pos, eye_center)) # opposite so that points will rotate opposite direction
    theta = np.arccos(my_dot(normalize(eye_pos), eye_center))
    return rot_axes, theta

def sphere2cartesian(azimuth, elevation, R=1):
    # In cartesian coordinates:
    x = R*np.sin((np.deg2rad(elevation)))*np.cos(np.deg2rad(azimuth))
    y = R*np.sin(np.deg2rad(elevation))*np.sin(np.deg2rad(azimuth))
    z = R*np.cos(np.deg2rad(elevation))
    return np.column_stack([x,y,z])

def calc_euler_params(axes, theta):
    a = np.cos(theta / 2)[:,None]
    w = axes * np.sin(theta / 2)[:,None]
    return a, w

def calc_rodrigues_rotation(a,w,x):
    w_x = np.cross(w,x)
    x_rot = x + 2*a*w_x + 2*np.cross(w, w_x)
    return x_rot

def eulerRodriguesVectorRotation(axes, theta, x):
    axes = normalize(axes)
    a = np.cos(theta / 2)[:,None]
    w = -axes * np.sin(theta / 2)[:,None]
    w_x = np.cross(w,x)
    x_new = x + 2*a*w_x + 2*np.cross(w, w_x)
    return x_new

def cartesian2retinal(points, R=1):
    x,y,z = points.T

    lat = np.arcsin(z / R)
    lon = np.arctan2(-y, x)

    eccentricity = 90 - np.rad2deg(lat) # azimuth
    polar = 360 - np.rad2deg(lon) - 360 # polar or elevation 

    return eccentricity, polar
    

def dome2eye(dome_x, dome_y, irec_x,irec_y, eye_coords):
    # check inputs shapes
    if dome_x.shape[0] != irec_x.size:
        raise ValueError('dimension 0 of dome_x %i and irec %i mismatch'%(dome_x.shape[0], irec_x.size))
    
    if dome_x.ndim > 1:
        npoints = dome_x.shape[1]
    else:
        npoints = 1

    # get locations on an eye unit sphere
    x,y,z = normalize(dome2cartesian(dome_x.ravel(), dome_y.ravel())-eye_coords).T # from eye to dome

    # shape them correctly
    dome_direction = np.zeros((dome_x.shape[0], 3, npoints))
    dome_direction[:,0,:] = x.reshape(-1,npoints)
    dome_direction[:,1,:] = y.reshape(-1,npoints)
    dome_direction[:,2,:] = z.reshape(-1,npoints)

    # rotate points opposite to eye direction
    rot_axes, theta = calc_irec_rotation(irec_x, irec_y)
    a, w = calc_euler_params(rot_axes, theta)

    eccentriciy = np.zeros(dome_x.shape)
    polar = np.zeros(dome_y.shape)
    for ipoint in range(npoints):
        rotated_coords = calc_rodrigues_rotation(a,w,dome_direction[:,:,ipoint])
        #print(rotated_coords)

        # convert to azimuth + elevation
        if npoints == 1:
            eccentriciy[:], polar[:] = cartesian2retinal(rotated_coords, R=1)
        else:
            eccentriciy[:,ipoint], polar[:,ipoint] = cartesian2retinal(rotated_coords, R=1)
    
    return eccentriciy, polar

