**Builds Status**

| **Windows** | **Mac** |
|---|---|
[![Build Status]()]()|[![Build Status]()]()

# OpenSRA
This repository contains the source code to OpenSRA_backend, developed by [Slate Geotechnical Consultants](http://slategeotech.com/), with assistance from [NHERI SimCenter](https://simcenter.designsafe-ci.org/) and [UC Berkeley](https://ce.berkeley.edu/).

## Developers
[Barry Zheng, PhD, @ Slate](http://slategeotech.com/)

[Steve Gavrilovic, PhD, @ NHERI SimCenter](https://simcenter.designsafe-ci.org/about/people/)

[Maxime Lacour, PhD, @ UC Berkeley](https://www.linkedin.com/in/maxime-lacour-637a8b79)

## Dependencies

### Python Packages
The Python packages required for the current version of OpenSRA are listed below, along with the versions used in development. Packages from the "conda-forge" channel were used. If experiencing difficulty with package installation, consider working in a clean environment.

[jpype1 (1.2.1)](https://jpype.readthedocs.io/en/latest/index.html) - please first install this package; incompatibility was observed when trying to install this package after the rest

[NumPy (1.19.5)](https://numpy.org/doc/stable/)

[pandas (1.2.1)](https://pandas.pydata.org/docs/)

[SciPy (1.6.0)](https://docs.scipy.org/doc/scipy/reference/)

[GeoPandas (0.8.1)](https://geopandas.org/) - dependency will be removed in the future

[rasterio (1.1.8)](https://rasterio.readthedocs.io/en/latest/) - dependency will be removed in the future

[shapely (1.7.1)](https://shapely.readthedocs.io/en/stable/manual.html) - dependency will be removed in the future

### Java
Currently, OpenSRA uses [OpenSHA](https://opensha.org/) for seismic source and ground motion characterizations. OpenSHA is developed in Java, and as such Java must be installed in order for OpenSRA to interface with OpenSHA. A procedure to install Java is provided below.

#### Windows 10
1. Download Java's Software Development Kit (JDK) from [https://jdk.java.net/](https://jdk.java.net/) (JDK 15 is used for current version of OpenSRA).
2. Install JDK on computer.
3. Navigate to "Control Panel".
4. Click "Systems".
5. On the left side panel, click "Advanced system settings".
6. Under the "Advanced" tab, click "Environment Variables".
7. Under "System Variables", if "JAVA_HOME" is not on the list, click "New".
8. Under "Variable name" enter "JAVA_HOME".
9. Under "Variable value" paste the path to JDK (e.g., C:\Program Files\Java\jdk-15.0.1).
10. To make sure Java is installed correctly, run "cmd", type in "java --version", and press enter. If a message that starts with "openjdk .." appears in the prompt, then a version of Java is installed. Confirm that the on-screen Java version is consistent with the downloaded version.

## User's Guide
To run OpenSRA in the command prompt, nagivate to the root folder of OpenSRA and run the command:
```
python OpenSRA.py -i PATH_TO_INPUT_FOLDER
```

To also clean outputs from the previous run, run the command:
```
python OpenSRA.py -i PATH_TO_INPUT_FOLDER -c yes
```

## Developer's Guide
Under development

## Acknowledgments
The OpenSRA development team would like to acknowledge [Dr. Wael Elhaddad and Dr. Kuanshi Zhong @ NHERI SimCenter](https://simcenter.designsafe-ci.org/about/people/) for providing developmental support on the OpenSHA interface, and [Dr. Simon Kwong @ USGS](https://www.usgs.gov/staff-profiles/neal-simon-kwong) for providing technical feedback on seismic and performance-based hazard analysis.

## License
Please check the license file in the root folder.
