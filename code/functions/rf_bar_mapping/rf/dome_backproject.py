import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RBFInterpolator

# def calculate_rotvec(direction):
#     x = np.cos(np.deg2rad(90-direction))
#     y = np.sin(np.deg2rad(90-direction))
#     return np.array([x,y,0])

def calculate_rotvec(vector, angles):
    axis = vector[-1,:] - vector[0,:]
    axis = axis / np.linalg.norm(axis)
    return angles[:,None]*axis[None,:]

def sphere2cartesian(azimuth, elevation, R=1):
    # In cartesian coordinates:
    x = R*np.sin((np.deg2rad(elevation)))*np.cos(np.deg2rad(azimuth))
    y = R*np.sin(np.deg2rad(elevation))*np.sin(np.deg2rad(azimuth))
    z = R*np.cos(np.deg2rad(elevation))
    return np.column_stack([x,y,z])

def dome_barprojection(direction, angles, steps=100):
    # return all bar points for all frames in a single direction
    nframes = len(angles)
    points = np.zeros((3, steps, nframes))

    azimuth_original = np.zeros(steps) #azimuth
    elevation_original = np.linspace(90, -90, steps) #elevation
    vector_orig = sphere2cartesian(90 - azimuth_original, 90 - elevation_original, R=1)

    direction_rot = Rotation.from_rotvec([0,0,-direction], degrees=True) # clockwise
    vector_rot = direction_rot.apply(vector_orig)

    sweep = Rotation.from_rotvec(calculate_rotvec(vector_rot, angles), degrees=True)

    for ii in range(len(angles)):
        points[:,:,ii] =  sweep[ii].apply(vector_rot).T

    return points

def halfdome_spherical_meshgrid(azimuth=[-90,90], elevation=[0,180], grid_deg=1):
    # grid size in degrees
    e = np.deg2rad(np.arange(elevation[0], elevation[1], grid_deg))
    a = np.deg2rad(np.arange(azimuth[0], azimuth[1], grid_deg))

    x = np.outer(np.sin(e), np.sin(a))
    y = np.outer(np.sin(e), np.cos(a))
    z = np.outer(np.cos(e), np.ones_like(a))

    return x, y, z

def halfdome_spherical_meshcenters(azimuth=[-90,90], elevation=[0,180], grid_deg=1):
    # grid size in degrees
    e = np.arange(elevation[0], elevation[1], grid_deg)[1:]-(grid_deg/2)
    a = np.arange(azimuth[0], azimuth[1], grid_deg)[1:]-(grid_deg/2)
    
    x_center = np.outer(np.sin(np.deg2rad(e)), np.sin(np.deg2rad(a)))
    y_center = np.outer(np.sin(np.deg2rad(e)), np.cos(np.deg2rad(a)))
    z_center = np.outer(np.cos(np.deg2rad(e)), np.ones_like(a))
    
    return x_center, y_center, z_center


def interpolate_activity_across_dome(points, activity):
    interpolator = RBFInterpolator(points, activity)

    interpolated_points = halfdome_spherical_meshgrid()
    smoothed_activity = interpolator(interpolated_points)

    return interpolated_points, smoothed_activity


def eulerRodriguesRotation(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(np.deg2rad(theta) / 2)
    b, c, d = -axis[:,None] * np.sin(np.deg2rad(theta) / 2)[None,:]
    a2, b2, c2, d2 = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [a2 + b2 - c2 - d2, 2*(bc - ad),       2*(bd + ac)],
        [2*(bc + ad),       a2 + c2 - b2 - d2, 2*(cd - ab)],
        [2*(bd - ac),       2*(cd + ab),       a2 + d2 - b2 - c2]
    ])


def barSweep(angles, vector_tilted):
    axis = vector_tilted[-1,:] - vector_tilted[0,:]
    RotMat = eulerRodriguesRotation(axis, angles)
    return np.dot(vector_tilted, RotMat)

def get_sweep_direction_points(direction, angles, steps):

    # We begin by generating a simple semi circle, aligned with one of the axes for simplicity.
    theta_original = np.linspace(0, 180, steps) # elevation
    phi_original =  90*np.ones((steps)) # azimuth
    vector_1 = sphere2cartesian(phi_original, theta_original, R=1)

    # We then tilt it around the dome vertical axis. This is the vector we will actually sweep.
    RotMat = eulerRodriguesRotation(np.array([0,1,0]), np.array((direction,direction)))
    vector_tilted = np.dot(vector_1, RotMat[:,:,0])

    #For the inclined bar (with an arbitrary angle around the screen center), sweep it by a given amount.
    sweep_points = barSweep(angles, vector_tilted) # shape= nsteps x 3 x nangles
    sweep_points = np.hstack(sweep_points.T).T # shape= (nstepsxnangles) x 3
    return sweep_points

# latitude/longitude calculations
# http://www.movable-type.co.uk/scripts/latlong.html

def bear(latA, lonA, latB, lonB):
    # BEAR Finds the bearing from one lat / lon point to another.
    y = np.sin(lonB - lonA) * np.cos(latB)
    x = np.cos(latA) * np.sin(latB) - np.sin(latA) * np.cos(latB) * np.cos(lonB - lonA)
    
    return np.arctan2(y,x)

def distance(latA, lonA, latB, lonB, R=1):
    #DIS Finds the distance between two lat/lon points
    return np.arccos( np.sin(latA)*np.sin(latB) + np.cos(latA)*np.cos(latB)*np.cos(lonB-lonA) ) * R


def pointToLineDistance(lon1, lat1, lon2, lat2, lon3, lat3, R=1):
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lat3 = np.deg2rad(lat3)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lon3 = np.deg2rad(lon3)
    

    bear12 = bear(lat1, lon1, lat2, lon2)
    bear13 = bear(lat1, lon1, lat3, lon3)
    dis13 = distance(lat1, lon1, lat3, lon3)
    
    bear_diff = bear13 - bear12
    
    if np.any(bear_diff > np.pi):
        chg = bear_diff > np.pi
        bear_diff[chg] = 2*np.pi - bear_diff[chg]

    # Find the cross-track distance.
    dxt = np.arcsin(np.sin(dis13 / R) * np.sin(bear_diff)) * R
    
    return np.abs(dxt)

def dist_grid2sweeps(fixed, sweeps, grid):
    # expecting (longitude, latitude) in degrees, nsweepsx2, ngridpointsx2
    # returns dist= nsweeps x ngridpoints
    
    # convert to radians
    fixed, sweeps, grid = np.deg2rad(fixed), np.deg2rad(sweeps), np.deg2rad(grid)
    
    # do calculations
    sweep_bear = bear(fixed[1],fixed[0], sweeps[:,1], sweeps[:,0])
    grid_bear = bear(fixed[1],fixed[0], grid[:,1], grid[:,0])
    fixedgrid_dist = distance(fixed[1],fixed[0], grid[:,1], grid[:,0])
    
    bear_diff = grid_bear[None,:] - sweep_bear[:,None]
    
    dist = np.arcsin(np.sin(fixedgrid_dist[None,:]) * np.sin(bear_diff))
    
    return np.abs(dist)

def cartesian2geo(cart, R=1):
    x,y,z = cart[:,0], cart[:,1], cart[:,2]
    
    phi = np.arccos(z/R)
    theta = np.arctan2(y,x)

    lon = np.rad2deg(theta) # azimuth
    lat = 90 - np.rad2deg(phi) # # polar or elevation 

    return lon, lat