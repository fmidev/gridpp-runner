A simple script to run gridpp program ([https://github.com/metno/gridpp]())
in FMI environments.

Note: requires access to Smartmet server not available outside FMI
networks.

# Installation

```
python3 -m pip install -r requirements.txt
```
Or

```
podman build -t gridpp-runner .
```

# Usage

```
usage: run.py [-h] --topography_data TOPOGRAPHY_DATA --landseacover_data
              LANDSEACOVER_DATA --parameter_data PARAMETER_DATA --parameter
              PARAMETER [--dem_data DEM_DATA] --output OUTPUT [--plot]
run.py: error: the following arguments are required: --topography_data, --landseacover_data, --parameter_data, --parameter, --output
```

For example:

```
python3 run.py --parameter temperature \
               --topography_data mnwc-Z-M2S2.grib2 \
               --parameter_data mnwc-T-K.grib2 \
               --landseacover_data mnwc-LC-0TO1.grib2 \
               --output T-K.grib2
```

DEM data can be downloaded from here:

[https://lake.fmi.fi/dem-data/DEM_100m-Int16.tif]()

To read/write from/to s3 directly, add `s3://` prefix to files.

Supports only temperature (K), humidity (%) and windspeed (m/s).
