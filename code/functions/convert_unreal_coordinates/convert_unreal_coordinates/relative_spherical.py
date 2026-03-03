import numpy as np

# cartesian Unreal coordinates relative to the player (x=forwards, y=sideways, z=vertical)
# spherical dome coordinates (azimuth=sideways, elevation=vertical, radius=radial distance) 

def spherical2relative(azimuth, elevation, radius):
    #assumes deg
    
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    
    x = (np.cos(azimuth) * np.cos(elevation)) * radius
    y = (np.sin(azimuth) * np.cos(elevation)) * radius
    z = np.sin(elevation) * radius
    return x, y, z

def relative2spherical(x, y, z):
    # outputs degrees
    
    radius = np.sqrt((x**2 + y**2 + z**2))
    azimuth = np.arctan2(y, x)
    elevation = np.arctan(np.sqrt(x**2 + y**2) / z)

    if np.any(elevation > 0):
        elevation[elevation > 0] = (np.pi / 2) - elevation[elevation > 0]
    elif np.any(elevation < 0):
        elevation[elevation < 0] = (-np.pi / 2) - elevation[elevation < 0]
        
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)
            
    return azimuth, elevation, radius

def find_stimulus_corners(azimuth, elevation, radius, width = 100, height = 100, vertical_offset = 0):
    '''
    Returns the spherical coordinates of the four corners of a stimulus, given its spawn location, scale and vertical offset (as provided by the parsing example below):

    ```
    with TextLog(filename) as log:        
        indx = [ii for ii, name in enumerate(log.all_ids['name']) if name.startswith(stim_name)]
        for ii, istim in enumerate(indx):
            if ii + n_stimuli == len(indx):
                break
            this_id = log.all_ids[istim]
            next_id = log.all_ids[indx[ii + n_stimuli]] # this should be the spawn time of the image in the next trial
            loc, pos_ts = log.parse_spherical(obj_id=this_id['id'],
                                                st=this_id['start'],
                                                end=next_id['start'])
            params = log.parse_initial_parameters(obj_id = this_id['id'], st = this_id['start'], end = next_id['start'])

            stim_loc.append(loc)
            stim_ts.append(pos_ts)
            stim_params.append(params)

    vertical_offset = np.append(arr = vertical_offset, values = [np.uint16(stimulus['Height']) for i, stimulus in enumerate(stim_params)])
    stim_width = np.append(arr = stim_width, values = [np.uint16(stimulus['Scale'] * 100) for i, stimulus in enumerate(stim_params)]) # In Unreal units
    ```
    '''
    
    xorig, yorig, zorig = spherical2relative(azimuth, elevation, radius)
    
    a0, e0, r = relative2spherical(xorig, yorig + width / 2, zorig + vertical_offset) #bottom right
    a1, e1, r = relative2spherical(xorig, yorig - width / 2, zorig + vertical_offset) #bottom left
    a2, e2, r = relative2spherical(xorig, yorig + width / 2, zorig + height + vertical_offset) #top right
    a3, e3, r = relative2spherical(xorig, yorig - width / 2, zorig + height + vertical_offset) #top left
    
    a = np.stack((a0, a2, a3, a1))  #br-tr-tl-bl
    e = np.stack((e0, e2, e3, e1))  #br-tr-tl-bl
    
    return a,e
