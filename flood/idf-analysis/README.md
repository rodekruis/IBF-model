# Goal
Knowing amount of precipitation falling down to an area can be used as an early warning. This is characterised by the intensity-duration-frequency (IDF) of rainfall. This task is to analyse IDF in southern Malawi using ECMWF ERA5 rainfall from 2001-2020. The goal is to derive IDF curves per traditional authority (admin level 3) for 1-72 hrs at 2, 5, 10, 20, 50 and 100 year return period.

# Prerequisites

## Data
The script was tested with the following files in the data folder:
- malawi-era5-malawi-rainfall-2001-2010.nc
- malawi-era5-malawi-rainfall-2011-2020.nc
- mwi_admbnda_adm3_nso_20181016_new.shp

## Packages
Some packages are needed to use this script. They can be installed using:
```
pip install -r requirements.txt
```
On M1 Macbooks this currently doesn't work, another solution is:
```
conda install -f environment.yml
conda activate idf
```

There were some small issues with the IDF library. If this PR is merged the requirements.txt and environment.yml can be updated to load from pypi again:
https://github.com/MarkusPic/intensity_duration_frequency_analysis/pull/5

# Usage
To run the script with default settings:
```bash
python idf.py
```

To check the arguments and how to use the script, run:
```bash
python idf.py --help
```

To use the results as a pandas df directly:
```python
import idf

df = idf.idf()
```
