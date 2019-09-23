"""
ground_track.py

Generate a GeoJSON of the Globals ground tracks for given interval from up-to-date TLEs. Ground track has been verified
against STK SGP4 propagator to < 1m.

TO-DO:
- include observable ground swath
"""
import datetime
import json
import re
import argparse

import numpy as np
import requests

from skyfield.api import load, utc
from skyfield.sgp4lib import EarthSatellite
from sunposition import sunpos

GLOBAL_CATALOG_NUMBER = {"G1": 43730, "G2": 43812, "G3": 44367, "G4": 44499}
CELESTRAK_URL = "https://www.celestrak.com/NORAD/elements/active.txt"


def create_ground_track(out_path, sat, tle=None, start=None, delta_min=1, num_steps=1440, min_sun=None):
    """
    Compute the ground track for Globals and then write to GeoJSON
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

    sun_zen = sunpos(times, lonlat[:,1], lonlat[:,0], 0)[:,1]
    sun_elev = 90 - sun_zen

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

    return track_list_out


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
    sat_id = GLOBAL_CATALOG_NUMBER[sat]
    regex1 = re.compile("1 {:}.*".format(sat_id))
    regex2 = re.compile("2 {:}.*".format(sat_id))

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

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if (args.line1 is None) or (args.line2 is None):
        tle = None
    else:
        tle = [args.line1, args.line2]

    create_ground_track(args.out_path, args.sat, tle, args.start, args.step_size, args.step_num, args.min_sun)
