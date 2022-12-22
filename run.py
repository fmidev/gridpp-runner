import gridpp
import numpy as np
import eccodes as ecc
import sys
import pyproj
import requests
import datetime
import argparse
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import fsspec
import os


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--parameter_data", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument(
        "--dem_data", action="store", type=str, default="DEM_100m-Int16.tif"
    )
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)

    args = parser.parse_args()

    allowed_params = ["temperature", "humidity", "windspeed"]
    if args.parameter not in allowed_params:

        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args


def get_shapeofearth(gh):
    """Return correct shape of earth sphere / ellipsoid in proj string format.
    Source data is grib2 definition.
    """

    shape = ecc.codes_get_long(gh, "shapeOfTheEarth")

    if shape == 1:
        v = ecc.codes_get_long(gh, "scaledValueOfRadiusOfSphericalEarth")
        s = ecc.codes_get_long(gh, "scaleFactorOfRadiusOfSphericalEarth")
        return "+R={}".format(v * pow(10, s))

    if shape == 5:
        return "+ellps=WGS84"


def get_falsings(projstr, lon0, lat0):
    """Get east and north falsing for projected grib data"""

    ll_to_projected = pyproj.Transformer.from_crs("epsg:4326", projstr)
    return ll_to_projected.transform(lat0, lon0)


def get_projstr(gh):
    """Create proj4 type projection string from grib metadata" """

    projstr = None

    proj = ecc.codes_get_string(gh, "gridType")
    first_lat = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
    first_lon = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")

    if proj == "polar_stereographic":
        projstr = "+proj=stere +lat_0=90 +lat_ts={} +lon_0={} {} +no_defs".format(
            ecc.codes_get_double(gh, "LaDInDegrees"),
            ecc.codes_get_double(gh, "orientationOfTheGridInDegrees"),
            get_shapeofearth(gh),
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    elif proj == "lambert":
        projstr = (
            "+proj=lcc +lat_0={} +lat_1={} +lat_2={} +lon_0={} {} +no_defs".format(
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin2InDegrees"),
                ecc.codes_get_double(gh, "LoVInDegrees"),
                get_shapeofearth(gh),
            )
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    else:
        print("Unsupported projection: {}".format(proj))
        sys.exit(1)

    return projstr


def read_file_from_s3(grib_file):
    uri = "simplecache::{}".format(grib_file)

    return fsspec.open_local(
        uri, s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}}
    )


def read_grib(gribfile, read_coordinates=False):
    """Read first message from grib file and return content.
    List of coordinates is only returned on request, as it's quite
    slow to generate.
    """

    wrk_gribfile = gribfile

    if gribfile.startswith("s3://"):
        wrk_gribfile = read_file_from_s3(gribfile)

    with open(wrk_gribfile) as fp:
        print("Reading {}".format(gribfile))
        gh = ecc.codes_grib_new_from_file(fp)

        ni = ecc.codes_get_long(gh, "Nx")
        nj = ecc.codes_get_long(gh, "Ny")
        dataDate = ecc.codes_get_long(gh, "dataDate")
        dataTime = ecc.codes_get_long(gh, "dataTime")
        forecastTime = ecc.codes_get_long(gh, "forecastTime")

        analysistime = datetime.datetime.strptime(
            "{}.{:04d}".format(dataDate, dataTime), "%Y%m%d.%H%M"
        )
        forecasttime = analysistime + datetime.timedelta(hours=forecastTime)

        values = ecc.codes_get_values(gh).reshape(nj, ni)

        if read_coordinates == False:
            return None, None, values, analysistime, forecasttime

        projstr = get_projstr(gh)

        di = ecc.codes_get_double(gh, "DxInMetres")
        dj = ecc.codes_get_double(gh, "DyInMetres")

        lons = []
        lats = []

        proj_to_ll = pyproj.Transformer.from_crs(projstr, "epsg:4326")

        for j in range(nj):
            y = j * dj
            for i in range(ni):
                x = i * di

                lat, lon = proj_to_ll.transform(x, y)
                lons.append(lon)
                lats.append(lat)

        return (
            np.asarray(lons).reshape(nj, ni),
            np.asarray(lats).reshape(nj, ni),
            values,
            analysistime,
            forecasttime,
        )


def read_grid(args):
    """Top function to read all gridded data"""

    lons, lats, vals, analysistime, forecasttime = read_grib(args.parameter_data, True)

    _, _, topo, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _ = read_grib(args.landseacover_data, False)

    grid = gridpp.Grid(lats, lons, topo)
    return grid, lons, lats, vals, analysistime, forecasttime


def read_conventional_obs(args, obstime):
    parameter = args.parameter

    timestr = obstime.strftime("%Y%m%d%H%M%S")

    trad_obs = []

    lons = []
    lats = []

    # conventional obs are read from two distinct smartmet server producers
    # if read fails, abort program

    for producer in ["observations_fmi", "foreign"]:
        url = "http://smartmet.fmi.fi/timeseries?producer={}&tz=gmt&precision=auto&starttime={}&endtime={}&param=fmisid,longitude,latitude,utctime,elevation,{}&format=json&data_quality=1&keyword=snwc".format(
            producer, timestr, timestr, parameter
        )

        resp = requests.get(url)

        # print(url)

        if resp.status_code != 200:
            print("Error")
            sys.exit(1)

        trad_obs += resp.json()

    obs = pd.DataFrame(trad_obs)
    obs = obs.rename(columns={"fmisid": "station_id"})

    count = len(trad_obs)

    print("Got {} traditional obs stations".format(count))

    if count == 0:
        print("Unable to proceed")
        sys.exit(1)

    return obs


def read_netatmo_obs(args, obstime):
    url = "http://smartmet.fmi.fi/timeseries?producer=NetAtmo&tz=gmt&precision=auto&starttime={}&endtime={}&param=station_id,longitude,latitude,utctime,temperature&format=json&data_quality=1&keyword=snwc".format(
        (obstime - datetime.timedelta(minutes=10)).strftime("%Y%m%d%H%M%S"),
        obstime.strftime("%Y%m%d%H%M%S"),
    )

    resp = requests.get(url)

    # print(url)

    crowd_obs = None

    if resp.status_code != 200:
        print("Error fetching NetAtmo data")
    else:
        crowd_obs = resp.json()

    print("Got {} crowd sourced obs stations".format(len(crowd_obs)))

    obs = None

    if crowd_obs is not None:
        obs = pd.DataFrame(crowd_obs)

        # netatmo obs do not contain elevation information, but we need thatn
        # to have the best possible result from optimal interpolation
        #
        # use digital elevation map data to interpolate elevation information
        # to all netatmo station points

        print("Interpolating elevation to NetAtmo stations")
        dem = xr.open_rasterio(args.dem_data)

        # dem is projected to lambert, our obs data is in latlon
        # transform latlons to projected coordinates

        ll_to_proj = pyproj.Transformer.from_crs("epsg:4326", dem.attrs["crs"])
        xs, ys = ll_to_proj.transform(obs["latitude"], obs["longitude"])
        obs["x"] = xs
        obs["y"] = ys

        # interpolated dem data to netatmo station points in x,y coordinates

        demds = dem.to_dataset("band").rename({1: "dem"})
        x = demds["x"].values

        # RegularGridInterpolator requires y axis value to be ascending -
        # geotiff is always descending

        y = np.flip(demds["y"].values)
        z = np.flipud(demds["dem"].values)

        interp = RegularGridInterpolator(points=(y, x), values=z)

        points = np.column_stack((obs["y"], obs["x"]))
        obs["elevation"] = interp(points)

        obs = obs.drop(columns=["x", "y"])

    return obs


def read_obs(args, obstime):
    """Read observations from smartmet server"""

    obs = read_conventional_obs(args, obstime)

    # for temperature we also have netatmo stations
    # these are optional

    if args.parameter == "temperature":
        netatmo = read_netatmo_obs(args, obstime)
        if netatmo is not None:
            obs = pd.concat((obs, netatmo))

        obs["temperature"] += 273.15

    points = gridpp.Points(
        obs["latitude"].to_numpy(),
        obs["longitude"].to_numpy(),
        obs["elevation"].to_numpy(),
    )

    return points, obs


def write_grib_message(fp, args, analysistime, forecasttime, data):

    levelvalue = 2
    pnum = 0
    pcat = 0

    if args.parameter == "humidity":
        pcat = 1
        pnum = 1
    elif args.parameter == "windspeed":
        pcat = 2
        pnum = 1
        levelvalue = 10

    h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ecc.codes_set(h, "gridType", "lambert")
    ecc.codes_set(h, "shapeOfTheEarth", 5)
    ecc.codes_set(h, "Nx", data.shape[1])
    ecc.codes_set(h, "Ny", data.shape[0])
    ecc.codes_set(h, "DxInMetres", 2370000 / (data.shape[1] - 1))
    ecc.codes_set(h, "DyInMetres", 2670000 / (data.shape[0] - 1))
    ecc.codes_set(h, "jScansPositively", 1)
    ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
    ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
    ecc.codes_set(h, "Latin1InDegrees", 63.3)
    ecc.codes_set(h, "Latin2InDegrees", 63.3)
    ecc.codes_set(h, "LoVInDegrees", 15)
    ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
    ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
    ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
    ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
    ecc.codes_set(
        h, "forecastTime", int((forecasttime - analysistime).total_seconds() / 3600)
    )
    ecc.codes_set(h, "centre", 86)
    ecc.codes_set(h, "generatingProcessIdentifier", 203)
    ecc.codes_set(h, "discipline", 0)
    ecc.codes_set(h, "parameterCategory", pcat)
    ecc.codes_set(h, "parameterNumber", pnum)
    ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
    ecc.codes_set(h, "scaledValueOfFirstFixedSurface", levelvalue)
    ecc.codes_set(h, "packingType", "grid_ccsds")
    ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 0)
    ecc.codes_set(h, "forecastTime", 0)
    ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
    ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products
    ecc.codes_set_values(h, data.flatten())
    ecc.codes_write(h, fp)
    ecc.codes_release(h)


def write_grib(args, analysistime, forecasttime, data):

    if args.output.startswith("s3://"):
        openfile = fsspec.open(
            "simplecache::{}".format(args.output),
            "wb",
            s3={
                "anon": False,
                "key": os.environ["S3_ACCESS_KEY_ID"],
                "secret": os.environ["S3_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
            },
        )
        with openfile as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
    else:
        with open(args.output, "wb") as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)

    print(f"Wrote file {args.output}")


def interpolate(grid, points, background, obs, args):
    """Perform optimal interpolation"""

    # Interpolate background data to observation points

    pobs = gridpp.nearest(grid, points, background)

    # Barnes structure function with horizontal decorrelation length 100km,
    # vertical decorrelation length 200m

    structure = gridpp.BarnesStructure(10000, 200)

    # Include at most this many observation points when interpolating to a grid point

    max_points = 20

    # error variance ratio between observations and background
    # smaller values -> more trust to observations

    obs_to_background_variance_ratio = np.full(points.size(), 0.01)

    # perform optimal interpolation

    print("Performing optimal interpolation")
    output = gridpp.optimal_interpolation(
        grid,
        background,
        points,
        obs[args.parameter].to_numpy(),
        obs_to_background_variance_ratio,
        pobs,
        structure,
        max_points,
    )

    return output


def main():
    args = parse_command_line()

    # Read required background data from files:
    # * land sea mask
    # * topography
    # * actual payload data

    print("Reading background data")
    grid, lons, lats, background, analysistime, forecasttime = read_grid(args)

    # Read observations from smartmet server

    print("Reading observation data")
    points, obs = read_obs(args, forecasttime)

    # Perform interpolation

    output = interpolate(grid, points, background, obs, args)

    write_grib(args, analysistime, forecasttime, output)

    if args.plot:
        plot(obs, background, output, lons, lats, args)


def plot(obs, background, output, lons, lats, args):
    import matplotlib.pyplot as plt

    vmin = min(np.min(background), np.min(output), np.min(obs[args.parameter]))
    vmax = min(np.max(background), np.max(output), np.max(obs[args.parameter]))

    plt.figure(figsize=(13, 6), dpi=80)

    plt.subplot(1, 3, 1)
    plt.pcolormesh(
        np.asarray(lons),
        np.asarray(lats),
        background,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(label="temperature (background)", orientation="horizontal")

    plt.subplot(1, 3, 2)
    plt.scatter(
        obs["longitude"],
        obs["latitude"],
        s=10,
        c=obs[args.parameter],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(label="temperature (points)", orientation="horizontal")

    plt.subplot(1, 3, 3)
    plt.pcolormesh(
        np.asarray(lons), np.asarray(lats), output, cmap="RdBu_r", vmin=vmin, vmax=vmax
    )

    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(
        label="temperature (optimal interpolation)", orientation="horizontal"
    )

    plt.show()


if __name__ == "__main__":
    main()
