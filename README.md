**Builds Status**

| **Windows** | **Mac** |
|---|---|
[![Build Status]()]()|[![Build Status]()]()

# OpenSRA
This repository contains the source code to OpenSRA_backend, developed by [Slate Geotechnical Consultants](http://slategeotech.com/), with assistance from [SimCenter@DesignSafe](https://simcenter.designsafe-ci.org/) and [UC Berkeley](https://ce.berkeley.edu/).

## Dependencies

### Python Packages
The Python packages required for the current version of OpenSRA are listed below, along with the versions used in development. Packages from the "conda-forge" channel were used. If experiencing difficulty with package installation, consider working in a clean environment.

jpype1 (1.2.1) - please first install this package first; incompatibility was observed when trying to install this package after the rest
NumPy (1.19.5)
pandas (1.2.1)
SciPy (1.6.0)

GeoPandas (0.8.1) - dependency will be removed in the future
rasterio (1.1.8) - dependency will be removed in the future
shapely (1.7.1) - dependency will be removed in the future

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
The OpenSRA development team would like to acknowledge Wael Elhaddad (previously SimCenter) and Kuanshi Zhong (Stanford University) for providing developmental support on the OpenSHA interface, and Smon Kwong (USGS) for providing technical feedback on seismic and performance-based hazard analysis.

## License
Please check the license file in the root folder.
