import utm
import numpy as np

def global_to_local(global_position, global_home):
    """
    Inputs: global_position: numpy array [longitude, latitude, altitude]; global_home: numpy array: [longitude, latitude, 
        altitude]
    Returns: local_position numpy array [northing, easting, altitude] of global_position
        relative to global_home [longitude, latitude, altitude]; Essentially, returns a
        numpy array describing the NED delta from global_home.
    """
    # Get easting and northing of global home first
    lonh, lath, alth = global_home[0], global_home[1], global_home[2]
    (easting_h, northing_h, zone_number_h, zone_letter_h) = utm.from_latlon(lath, lonh)
    
    # Get easting and northing from global position next
    lon, lat, alt = global_position[0], global_position[1], global_position[2]
    (easting, northing, zone_number, zone_letter) = utm.from_latlon(lat, lon)
    
    # After that, create a local_position numpy array from its NED coordinates
    local_position = np.array([northing - northing_h, easting - easting_h, -(alt - alth)])
    
    # Finally, return them.
    return local_position


def local_to_global(local_position, global_home):
    """
    Inputs: local_position: numpy array [northing, easting, down]; global_home: numpy array [longitude, latitude, altitude]
    Returns: global_position: numpy array [longitude, latitude, altitude]. Essentially, returns a numpy array holding a new
        geodetic location given a home geodetic location and an NED delta. 
    """
    # First, get global_home's easting, northing, zone_number, and grid_letter.
    lon_h, lat_h, alt_h = global_home[0], global_home[1], global_home[2]
    (easting_h, northing_h, zone_number_h, zone_letter_h) = utm.from_latlon(lat_h, lon_h)
    
    # After that, compute the new NED location by adding local_position to home's NED coordinates.
    northing, easting, alt = local_position[0], local_position[1], local_position[2] 
    
    # Convert the new NED location to the geodetic frame next. 
    (latitude, longitude) = utm.to_latlon(easting + easting_h, northing + northing_h, zone_number_h, zone_letter_h)
    altitude = -(alt - alt_h)
    global_position = np.array([longitude, latitude, altitude])
    
    # Finally, return the new geodetic location.
    return global_position