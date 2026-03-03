import numpy as np

# dome_x = np.array([0, 10, 0, -10, 0])
# dome_y = np.array([10, 0, 0, 0, -10])

# 14/4/22
# X = Right of center is positive
# Y = Above center is positive
# Z = In front of center is positive
# P_ref = np.array([37.5,-33.57,-33.17]) 
# P_eye = P_ref+np.array( [-36, 36.5, 19.4] )

# Input. From dome to iRec coordinates

def dome2cartesian(dome_x, dome_y, R=60):
    theta = np.deg2rad(90 - dome_x) # azimuth
    phi = np.deg2rad(90 - dome_y) # # polar or elevation 
    
    # In cartesian coordinates:
    x = R*np.sin((phi))*np.cos(theta)
    z = R*np.sin(phi)*np.sin(theta)
    y = R*np.cos(phi)
 
    return np.column_stack([x,y,z])

def point2plane(points, plane_distance):
    x,y,z = points[:,0], points[:,1], points[:,2]
    
    xprime = (x*(plane_distance-z))/z + x
    yprime = (y*(plane_distance-z))/z + y
    zprime = np.repeat(plane_distance, xprime.size)
    
    return np.column_stack((xprime, yprime, zprime))

def irec_flat_projection(points):
    x,y,z = points[:,0], points[:,1], points[:,2]
    x_angle = np.arctan(x/z)
    y_angle = np.arctan(y/z)

    return np.rad2deg(x_angle), np.rad2deg(y_angle)


# Output. From iRec to dome coordinates

def irec2cartesian(irec_x, irec_y, z):
    x = np.tan(np.deg2rad(irec_x))*z
    y = np.tan(np.deg2rad(irec_y))*z
    
    return np.column_stack((x,y,z))

def normalize(vector):
    return vector / np.linalg.norm(vector,axis=1)[:,None]
    
def angle2direction(eye_x, eye_y):
    
    alpha = np.deg2rad((eye_x-90)*-1)
    beta = np.deg2rad(eye_y)
    
    # In unit coordinates:
    x = np.cos((alpha))*np.cos(beta)
    z = np.sin(alpha)*np.cos(beta)
    y = np.sin(beta)
    
    return np.column_stack((x,y,z))

def sphere_intersect(ray_origin, ray_direction, R=60):
    b = 2*np.inner(ray_direction,ray_origin)
    c = np.linalg.norm(ray_origin)**2 - R**2
    delta = b**2 - 4*c
    
    t1 = (-b + np.sqrt(delta))/2
    t2 = (-b - np.sqrt(delta))/2
    if np.any(t1+t2 == 0):
        raise ValueError('Sphere is unexpectedly not intersected')
    else:
        return ray_origin + np.maximum(t1,t2)[:,None]*ray_direction

def cartesian2dome(dome_points, R=60): # z=y y=z
    x,y,z = dome_points[:,0], dome_points[:,1], dome_points[:,2]
    
    theta = np.arctan2(z,x)
    phi = np.arccos(y/R)

    dome_x = 90 - np.rad2deg(theta) # azimuth
    dome_y = 90 - np.rad2deg(phi) # # polar or elevation 

    return dome_x, dome_y

def eye2dome(irec_x, irec_y, eye_coords):
    irec_z = np.repeat(60-eye_coords[2], irec_x.size)
    irec_direction = normalize(irec2cartesian(irec_x, irec_y, irec_z))
    new_dome_coordinates = sphere_intersect(eye_coords, irec_direction)
    
    dome_x, dome_y = cartesian2dome(new_dome_coordinates)
    return dome_x, dome_y
