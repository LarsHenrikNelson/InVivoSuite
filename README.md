# InVivoSuite
InVivoSuite is intended for analysis LFPs recorded using multisite silicon probes and is currently a work in progress with major changes occuring. InVivoSuite can also convert Plexon AnalogChannels from pl2 files to hdf5 files as long as you use Windows since Plexon does not provide share libraries for Linux or Mac. This package is not necessarily intended for broad use but tailored for specific analysis requirements that I have for analyzing LFP recordings and interfacing with Kilosort and Phy analyzed data. The files data_analysis.py, data_processing.py and final_analysis.py are intended to show examples of how the package is used but are still under construction as I continue to work on analyzing the data so they may not be very informative at the moment.

If you want to use the package you will need to install the [Microsoft Visual Studio Compiler and C++ tools](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022).