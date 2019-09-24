"""
ground_track.py

Generate a GeoJSON of the Globals ground tracks for given interval from up-to-date TLEs. Ground track has been verified
against STK SGP4 propagator to < 1m.

TO-DO:
- add point at epoch
"""
import datetime
import json
import re
import argparse
import os
import random

import numpy as np
import requests

from skyfield.api import load, utc
from skyfield.sgp4lib import EarthSatellite
from pysolar.solar import get_altitude_fast as get_sun_elevation
import pymap3d

GLOBALS_CATALOG_NUMBER = {"G1": 43730, "G2": 43812, "G3": 44367, "G4": 44499}
GLOBALS_ALTITUDE_KM = {"G1": 490, "G2": 590, "G3": 460, "G4": 550}
EARTH_RAD_KM = 6371
DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi
CELESTRAK_URL = "https://www.celestrak.com/NORAD/elements/active.txt"


def create_ground_track(out_path, sat, tle=None, start=None, delta_min=1, num_steps=1440, min_sun=None, sensor_angle=None, check_swath=False):
    """
    Compute the ground track and sensor swatch for Globals and then write to GeoJSON
    """
    # get globals tles from Celestrak if TLE not passed in
    if tle is None:
        tle = get_globals_tle(sat)

    timescale = load.timescale(builtin=True)
    times_dt = generate_time_array(start, delta_min, num_steps)
    times = timescale.utc(times_dt)

    sat_obj = EarthSatellite(line1=tle[0], line2=tle[1])
    subsat = sat_obj.at(times).subpoint()

    lat = subsat.latitude.degrees
    lon = subsat.longitude.degrees

    track = np.concatenate([lon.reshape(-1, 1), lat.reshape(-1, 1)], axis=1)

    track_list = filter_sun_elevation(track, times_dt, min_sun)

    track_list = correct_rollover(track_list)

    properties = {"sat": sat, "start_time": str(times_dt[0]), "stop_time": str(times_dt[-1]), "time_step_minutes": delta_min}

    write_track_geojson(out_path, track_list, properties)

    if sensor_angle is not None:
        swath_list = get_sensor_swath(track_list, sensor_angle, GLOBALS_ALTITUDE_KM[sat])

        swath_out_path = os.path.splitext(out_path)[0] + "_swath" + os.path.splitext(out_path)[1]
        write_track_geojson(swath_out_path, swath_list, properties)

    if check_swath:
        swath_check_out_path = os.path.splitext(out_path)[0] + "_swath_test" + os.path.splitext(out_path)[1]
        check_swath(swath_check_out_path, track_list, sensor_angle, 1000*GLOBALS_ALTITUDE_KM[sat])


def check_swath(out_path, track_list, sensor_angle, altitude_m):
    """
    Compute individual sensor cones by intersecting line of sight with Earth at random track points. Only use to check
    that get_sensor_swath is working correctly
    """
    check_az = np.arange(0, 360, 5)

    check_list = []

    # pick 3 points from each path and check swath with pymap
    for track in track_list:
        for _ in range(3):
            lonlat = random.choice(track)

            check = []
            for az in check_az:
                lat, lon, _ = pymap3d.lookAtSpheroid(lonlat[1], lonlat[0], altitude_m, az, sensor_angle)
                check.append([lon[()], lat[()]])

            check_list.append(check)

    write_track_geojson(out_path, check_list, {"test": "test"})



def get_sensor_swath(track_list, sensor_angle, altitude):
    """
    Get a list of swath paths (list of lonlat pairs) for the sensor angle moving along each track at altitude
    """

    swath_list = []

    for track in track_list:
        bearings = get_track_bearings(track)

        swaths = compute_swaths(track, bearings, sensor_angle, altitude)

        swath_list.extend(swaths)

    return swath_list


def compute_swaths(track, bearings, sensor_angle, altitude):
    """
    Use the central angle computed from altitude and sensor angle to compute the lat/lon extent of the swath
    """
    central_angle = RAD2DEG * np.arcsin((EARTH_RAD_KM + altitude) / EARTH_RAD_KM * np.sin(DEG2RAD * sensor_angle)) - sensor_angle

    sin_central = np.sin(DEG2RAD * central_angle)
    cos_central = np.cos(DEG2RAD * central_angle)

    lonlat_track = np.array(track)
    heading1 = np.array(bearings) + 90
    heading2 = np.array(bearings) - 90

    sin_lat = np.sin(DEG2RAD * lonlat_track[:, 1])
    cos_lat = np.cos(DEG2RAD * lonlat_track[:, 1])

    sin_heading1 = np.sin(DEG2RAD * heading1)
    cos_heading1 = np.cos(DEG2RAD * heading1)
    sin_heading2 = np.sin(DEG2RAD * heading2)
    cos_heading2 = np.cos(DEG2RAD * heading2)

    swath1_lat = RAD2DEG * np.arcsin(sin_lat * cos_central + cos_lat * sin_central * cos_heading1)
    swath2_lat = RAD2DEG * np.arcsin(sin_lat * cos_central + cos_lat * sin_central * cos_heading2)

    swath1_lon = lonlat_track[:, 0] + RAD2DEG * np.arctan2(sin_heading1 * sin_central * cos_lat,
                                                           cos_central - sin_lat * np.sin(DEG2RAD * swath1_lat))
    swath2_lon = lonlat_track[:, 0] + RAD2DEG * np.arctan2(sin_heading2 * sin_central * cos_lat,
                                                           cos_central - sin_lat * np.sin(DEG2RAD * swath2_lat))

    swath1 = np.concatenate([swath1_lon.reshape(-1,1), swath1_lat.reshape(-1,1)], axis=1).tolist()
    swath2 = np.concatenate([swath2_lon.reshape(-1, 1), swath2_lat.reshape(-1, 1)], axis=1).tolist()

    return [swath1, swath2]


def get_track_bearings(track):
    """
    Get the average bearing at each point along the track, where average is the mean of incoming and outgoing segments
    """
    lonlat = np.array(track)

    # get the start and end points of each track segment
    start_lonlat = lonlat[:-1,:]
    end_lonlat = np.roll(lonlat, -1, axis=0)[:-1,:]

    segment_bearings = calc_bearings(start_lonlat, end_lonlat)

    bearings_out = segment_bearings
    bearings_in = np.roll(segment_bearings, 1)

    # correct incoming for first element
    bearings_in[0] = segment_bearings[0]

    # add in/out for the last point in track
    bearings_out = np.append(bearings_out, segment_bearings[-1])
    bearings_in = np.append(bearings_in, segment_bearings[-1])

    bearings_avg = (bearings_in + bearings_out) / 2

    return bearings_avg.tolist()


def calc_bearings(lonlat1, lonlat2):
    """
    Calculate the bearing from pt1 to pt2 at pt1. Each lonlat is Nx2
    """
    delta_lon = lonlat2[:,0] - lonlat1[:,0]

    sin_lon = np.sin(DEG2RAD * delta_lon)
    cos_lon = np.cos(DEG2RAD * delta_lon)

    sin_lat1 = np.sin(DEG2RAD * lonlat1[:,1])
    cos_lat1 = np.cos(DEG2RAD * lonlat1[:,1])

    sin_lat2 = np.sin(DEG2RAD * lonlat2[:,1])
    cos_lat2 = np.cos(DEG2RAD * lonlat2[:,1])

    return RAD2DEG * np.arctan2(sin_lon * cos_lat2, cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_lon)


def write_track_geojson(out_path, track_list, properties):
    """
    Write ground track to geojson
    """
    data = {"type": "Feature", "properties": properties,
            "geometry": {"type": "MultiLineString", "coordinates": track_list}}

    with open(out_path, "w") as fptr:
        json.dump(data, fptr)


def filter_sun_elevation(lonlat, times, min_sun):
    """
    Remove all lon/lat points where the sun elevation is below min threshold.
    """
    if min_sun is None:
        return [lonlat]

    sun_elev = np.array([get_sun_elevation(lat, lon, time) for lat, lon, time in zip(lonlat[:,1], lonlat[:,0], times)])

    this_is_sun = sun_elev >= min_sun
    last_is_sun = np.roll(this_is_sun, 1)
    last_is_sun[0] = False

    # get indices where we go from dark to light and light to dark
    start_track = np.argwhere((~last_is_sun & this_is_sun)).reshape(-1)
    end_track = np.argwhere((~this_is_sun & last_is_sun)).reshape(-1)

    # if last point in track is in sun, then
    if this_is_sun[-1]:
        end_track = np.append(end_track, len(this_is_sun))

    lonlat_tracks = []

    for start, end in zip(start_track, end_track):
        lonlat_tracks.append(lonlat[start:end, :])

    return lonlat_tracks


def correct_rollover(track_list):
    """
    Account for longitude rollover between +180/-180.
    Return list of tracks that each stay within [-180, +180], where is track is a nested list of lon/lat
    """
    track_list_out = []

    for track in track_list:
        lon = track[:,0]
        lon_sign = np.sign(lon)

        lon_signchange = (np.roll(lon_sign, 1) - lon_sign) != 0
        lon_signchange[0] = False

        lon_delta = np.abs(np.roll(lon, 1) - lon)

        rollover = (lon_delta > 180) & lon_signchange
        rollover_ind = np.argwhere(rollover)

        prev_idx = 0
        for idx in np.append(rollover_ind, track.shape[0]):
            track_list_out.append(track[prev_idx:idx, :].tolist())
            prev_idx = idx

    # remove lists with only one lat/lon
    track_list_prune = []
    for track in track_list_out:
        if len(track) > 1:
            track_list_prune.append(track)

    return track_list_prune


def generate_time_array(start=None, delta_min=1, num_steps=1440):
    """
    Return array of datetime objects with times to propagate
    """

    if start is None:
        start_dt = datetime.datetime.now()
    else:
        start_dt = parse_date(start)

    start_dt = start_dt.replace(tzinfo=utc)
    delta = datetime.timedelta(minutes=delta_min)

    return [start_dt + n * delta for n in range(num_steps)]


def parse_date(date_str):
    """
    Attempt to parse date with multiple formats
    """
    date_formats = ["%Y-%m-%d_%H-%M-%S",
                    "%Y%m%d%H%M%S",
                    "%Y-%m-%d",
                    "%Y%m%d"]

    for date_format in date_formats:
        try:
            return datetime.datetime.strptime(date_str, date_format)
        except ValueError:
            pass

    raise ValueError("Cannot parse date. Use format from {:}".format(date_formats))


def get_globals_tle(sat):
    """
    Get TLE for Globals
    """
    # get latest active TLEs
    response = requests.get(CELESTRAK_URL)
    data = response.text

    # search for TLE by ID
    sat_id = GLOBALS_CATALOG_NUMBER[sat]
    regex1 = re.compile("1 {:}.+".format(sat_id))
    regex2 = re.compile("2 {:} .+".format(sat_id))

    line1 = regex1.search(data)[0]
    line2 = regex2.search(data)[0]

    return [line1, line2]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("out_path", help="Path to geojson output file")
    p.add_argument("sat", help="Name of satellite. If not G1, G2, G3, G4 must pass in TLE")
    p.add_argument("--line1", default=None, help="First line of TLE to pass in")
    p.add_argument("--line2", default=None, help="Second line of TLE to pass in")
    p.add_argument("--start", default=None, help="Start time of groundtrack in format yyyy-mm-dd_hh-mm-ss, yyyymmddhhmmss, yyyy-mm-dd, or -yyyymmdd")
    p.add_argument("--step-size", default=1, type=float, help="Propagation time step size in minutes")
    p.add_argument("--step-num", default=1440, type=int, help="Number of propagation steps")
    p.add_argument("--min-sun", default=None, type=float, help="Minimum sun angle for plotting pass")
    p.add_argument("--sensor-angle", default=None, type=float, help="Half angle of the sensor swath to show")
    p.add_argument("--check-swath", action="store_true", help="Flag to generate GeoJSON to check swath width")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if (args.line1 is None) or (args.line2 is None):
        tle = None
    else:
        tle = [args.line1, args.line2]

    create_ground_track(args.out_path, args.sat, tle, args.start, args.step_size, args.step_num, args.min_sun,
                        args.sensor_angle, args.check_swath)
