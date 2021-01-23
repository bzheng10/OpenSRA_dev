**Builds Status**

| **Windows** | **Mac** |
|---|---|
[![Build Status]()]()|[![Build Status]()]()

# OpenSRA
This repository contains the source code to OpenSRA_backend, developed by [Slate Geotechnical Consultants](http://slategeotech.com/), with assistance from [NHERI SimCenter](https://simcenter.designsafe-ci.org/) and [UC Berkeley](https://ce.berkeley.edu/).

## Developers
Barry Zheng, PhD, @ Slate: http://slategeotech.com/

Steve Gavrilovic, PhD, @ NHERI SimCenter: https://simcenter.designsafe-ci.org/about/people/

Maxime Lacour, PhD, @ UC Berkeley: [LinkedIn](https://www.linkedin.com/in/maxime-lacour-637a8b79)

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
##### Option 1 - JDK (Oracle Distribution)
1. Download the Oracle distribution of Java at: https://www.oracle.com/java/technologies/javase-downloads.html
2. Once the installer is downloaded, follow the instructions here to install Java: https://docs.oracle.com/en/java/javase/15/install/installation-jdk-microsoft-windows-platforms.html#GUID-A7E27B90-A28D-4237-9383-A58B416071CA
3. Windows users need to manually set the path to the Java binary. Please follow the instructions in the link under step 2.
4. To make sure that Java is installed correctly, run Windows command prompt "cmd", type in "java --version", and press enter. If the prompt returns the correct Java version as the one installed, then Java is installed correctly.
5. If Step 4 fails, please, attemp Steps 5 through 16 under Option 2; if that fails also, then perform Option 2 in its entirety.

##### Option 2 - OpenJDK
###### Downloading Java
1. Go to the website for Java Software Development Kit (JDK) [https://jdk.java.net/](https://jdk.java.net/).
2. Click the link for the ready-to-use version of JDK download the "zip" file for Windows (JDK v15 was used in this version of OpenSRA)
3. Extract the zip file "openjdk-XX.Y.Z_windows-x64_bin.zip". The extracted folder should have the name "jdk-XX.YY.ZZ"
4. Create a folder named "Java" under "Program Files" and place the extracted folder "jdk-XX.YY.ZZ" into the "Java" directory.
###### Setting up Java on Windows
5. Open any folder and right click on "This PC" on the left panel.
6. Click "Properties".
7. On the left side panel of the pop-up window, click "Advanced system settings".
8. Under the "Advanced" tab, click "Environment Variables".
9. Under "System Variables", if "JAVA_HOME" is not on the list, click "New".
10. Under "Variable name" enter "JAVA_HOME".
11. Under "Variable value" paste the path to the JDK folder (e.g., C:\Program Files\Java\jdk-XX.YY.ZZ).
12. Click "OK" to save and get back to the "Environment Variables" pop-up window.
13. Find path under "System variables", or create a new variable if it doesn't exist.
14. Depending on the window that pops up:
	- If the pop-up window contains only two lines for input ("Variable name" and "Variable value"), add a semicolon (";") after the last entry in "Variable value" and append the text "%JAVA_HOME%\bin" (without the quotations).
	- If the pop-up window contains a list of entries on the left and a list of commands on the right, double click on the first empty row on the left side and paste in the string "%JAVA_HOME%\bin" as the value.
15. To make sure that Java is installed correctly, run Windows command prompt "cmd", type in "java --version", and press enter. If the prompt returns the correct Java version as the one installed, then Java is installed correctly.

#### MacOS
1. Go to the website for Java Software Development Kit (JDK) [https://jdk.java.net/](https://jdk.java.net/).
2. Once the installer is downloaded, follow the instructions here to install Java: https://docs.oracle.com/en/java/javase/15/install/installation-jdk-macos.html#GUID-2FE451B0-9572-4E38-A1A5-568B77B146DE

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
