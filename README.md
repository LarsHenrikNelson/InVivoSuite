# InVivoSuite
InVivoSuite is intended for analysis LFPs and spike data recorded using multisite silicon probes and is currently a work in progress with major changes occuring. InVivoSuite can also convert Plexon AnalogChannels from pl2 files to hdf5 files as long as you use Windows since Plexon does not provide libraries for Linux or Mac. InVivoSuite also can export a binary file for use in Kilosort and export Kilosort for use in Phy. The Phy export function works better than Phy's export-waveforms. This package is not necessarily intended for broad use but tailored for specific analysis requirements that I have for analyzing LFP recordings and interfacing with Kilosort and Phy analyzed data.

If you want to use the package you will need to install the [Microsoft Visual Studio Compiler and C++ tools](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022) if you are using Windows. For Mac install XCode and the associated commandline tools. For linux g++ should already be installed otherwise install g++/gcc.

For examples on how to use the package see the file-setup.py, spike-analysis.py and spike-lfp.py.