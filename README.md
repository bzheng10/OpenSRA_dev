<!---
**Builds Status**

| **Windows** | **Mac** |
|---|---|
[![Build Status]()]()|[![Build Status]()]()
-->

# OpenSRA
This repository contains the source code to OpenSRA_backend, developed by [Slate Geotechnical Consultants (Slate)](http://slategeotech.com/), with assistance from [NHERI SimCenter](https://simcenter.designsafe-ci.org/) and [UC Berkeley](https://ce.berkeley.edu/).

## Developers
Barry Zheng, PhD @ [Slate](https://slategeotech.com/people/)

Steve Gavrilovic, PhD @ NHERI SimCenter: [LinkedIn](https://www.linkedin.com/in/stevan-gavrilovic-berkeley/)

Maxime Lacour, PhD @ UC Berkeley: [LinkedIn](https://www.linkedin.com/in/maxime-lacour-637a8b79)

## Dependencies

### Python
OpenSRA has been tested on **Python version 3.9.9**
s
The Python modules required for the current version of OpenSRA are listed below, along with the versions used for testing. Modules were installed via "conda", the "conda-forge" channel, and "pip". If you are experiencing difficulty installing the modules, consider working in a clean environment.

From "conda-forge"

[numpy (1.20.3)](https://numpy.org/doc/stable/)

[pandas (1.3.5)](https://pandas.pydata.org/docs/)

[scipy (1.5.3)](https://docs.scipy.org/doc/scipy/reference/)

[openpyxl (3.0.9)](https://openpyxl.readthedocs.io/en/stable/)

[h5py (3.6.0)](https://www.h5py.org/)

[geopandas (0.10.2)](https://geopandas.org/)

[rasterio (1.2.10)](https://rasterio.readthedocs.io/en/latest/)

[shapely (1.8.0)](https://shapely.readthedocs.io/en/stable/manual.html)

[numba (0.54.1)](https://numba.pydata.org)

[icc_rt (2022.1.0)](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html)

[pygeos (0.12.0)](https://pygeos.readthedocs.io/en/stable/)

```
conda install -c conda-forge geopandas shapely pygeos openpyxl numba icc_rt h5py
```

From "pip"

[numba_stats (1.1.0)](https://github.com/HDembinski/numba-stats/)

[openquake.engine (3.14.0)](https://github.com/gem/oq-engine/)

[tables (3.7.0)](https://www.pytables.org/usersguide/installation.html/)

```
install numba_stats openquake.engine tables
```

## User's Guide
To run OpenSRA in the command prompt:

1. Nagivate to the root folder of OpenSRA.
2. Run preprocess command on the working directory using the command:
```
python Preprocess.py -w FULL_PATH_TO_INPUT_FOLDER
```
3. Once preprocessing is finished, run the main script for OpenSRA using the command:
To clean results from the previous run, run the command:
```
python OpenSRA.py -w FULL_PATH_TO_INPUT_FOLDER
```

## Developer's Guide
Under development

## Acknowledgements
The OpenSRA development team would like to acknowledge [Dr. Wael Elhaddad and Dr. Kuanshi Zhong @ NHERI SimCenter](https://simcenter.designsafe-ci.org/about/people/) for providing developmental support on the OpenSHA interface, and [Dr. Simon Kwong @ USGS](https://www.usgs.gov/staff-profiles/neal-simon-kwong) for providing technical feedback on seismic and performance-based hazard analysis.

## License
Please check the license file in the root folder.

