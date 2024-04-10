# %%
from pathlib import Path
import pandas as pd
from invivosuite.acq import AcqManager, load_pl2_acqs, load_hdf5_acqs

# %%
"""
Use this if creating hdf5 acquisitions for the first time.
"""
# pl2_paths = list(Path("linux/path/to/data").rglob("*.pl2"))
# save_path =  "linux/path/to/data"

pl2_paths = list(Path(r"C:\windows\path\to\data").rglob("*.pl2"))
save_path = r"C:\windows\path\to\data"

# %%
"""
If you want multiple save paths just create a list of filepaths
"""
acqs = []
for file_path in pl2_paths:
    acq_manager = load_pl2_acqs(str(file_path), save_path, end=None)
    acqs.append(acq_manager)


# %%
"""
There are two ways to load all you files at once.
Use get a list of files the manually load them
or you can use the convenience function load_hdf5_acqs.
Note that the package uses lazy loading so no files or
there content are load into memory. The file is just set as
an attribute of the AcqManager.
"""

# %%
"""
Create managers for all the files that you want to process.
"""
file_paths = list(Path(r"C:\windows\path\to\data").rglob("*.hdf5"))
# file_paths = list(Path("linux/path/to/data").rglob("*.hdf5"))
acqs = []
for file_path in file_paths:
    acq_manager = AcqManager()
    acq_manager.set_hdf5_file(file_path)
    acqs.append(acq_manager)

# %%
"""
Use convenience function to load the files
"""
# parent_path = r"C:\windows\path\to\data"
parent_path = "linux/path/to/data"
acqs = load_hdf5_acqs(parent_path)

# %%
"""
Set the filters for lfp and spike data using a zerophase butterworth filter
While the sample rate is needed to filter there is a sample rate per acquisition
that comes from the pl2 file so there is no need to supply it.
"""
for i in acqs:
    i.set_filter(
        acq_type="lfp",
        filter_type="butterworth_zero",
        order=4,
        highpass=0.5,
        lowpass=300,
        sample_rate=1000.0,
    )
    i.set_filter(
        acq_type="spike",
        filter_type="butterworth_zero",
        order=4,
        highpass=300,
        lowpass=5000,
    )

# %%
"""
Set the start end of the acquisitions if you want to analyze just a subset
of the acquisition. Useful if your aquisitions are different sizes between
recordings. The start and end are automatically set to the lenght of the
recoding which is pulled from the pl2 file.
"""
for i in acqs:
    i.set_start(0)
    i.set_end(40000 * 60 * 10)

# %%
"""
Set the groups for the channels since the .pl2 file contained both
ACC and DMS file. This attribute needs to be set to work with the spike
data.
"""
for i in acqs:
    i.set_probe("acc", [64, 128])
    i.set_probe("dms", [0, 64])


# %%
"""
Compute virtual reference (CMR or CAR)
"""
for i in acqs:
    i.compute_virtual_ref(ref_type="cmr", probe="acc", bin_size=0)
    # i.compute_virtual_ref(ref_type="car", probe="dms", bin_size=0)

# %%
"""
Load channel map
"""
chan_map = pd.read_csv(
    r"C:\Users\lhn6\OneDrive - University of Pittsburgh\exp_data\channel_maps\corr_map_correct.csv",
    header=None,
)
# Set the channel map
for i in acqs:
    i.set_channel_map(
        chan_map[0].to_numpy(),
        "acc",
    )
    i.set_channel_map(
        chan_map[0].to_numpy(),
        "dms",
    )

# %%
"""
Setting some random attributes
"""
for i in acqs:
    s_temp = i.file_path.stem.split("_")
    i.set_file_attr("date", f"{s_temp[0]}_{s_temp[1]}_{s_temp[2]}")
    i.set_file_attr("sex", s_temp[3])
    i.set_file_attr("id", s_temp[4])
    i.set_file_attr("genotype", s_temp[5])

# %%
"""
Find lfp bursts.
"""
for index, i in enumerate(acqs[18:]):
    print(index + 18)
    i.find_lfp_bursts(
        window="hamming",
        min_len=0.2,
        max_len=20.0,
        min_burst_int=0.2,
        wlen=0.2,
        threshold=2.0,
        pre=3.0,
        post=3.0,
        order=0.1,
        method="spline",
        tol=0.001,
        deg=30,
        cmr=True,
    )

# %%
"""
Set continuous wavelet properties
"""
for i in acqs:
    i.set_cwt(
        f0=1,
        f1=110,
        fn=400,
        scaling="log",
        norm=True,
        nthreads=-1,
    )

# %%
"""
Set Welch PSD properties
"""
for i in acqs:
    i.set_welch(nperseg=2048, noverlap=1000, window=("tukey", 0.25), scaling="density")

# %%
# I recommend the max_int method.
for index, i in enumerate(acqs):
    i.analyze_bursts(
        method="max_int",
        min_count=5,
        min_dur=0,
        max_start=0.170,
        max_end=0.300,
        max_int=0.200,
    )

# %%
"""
Calculate the Phase discountinuity index
"""
bands = {
    "theta": (4, 10),
    "low_gamma": (30, 50),
    "high_gamma": (70, 80),
    "beta": (12, 28),
}
for index, i in enumerate(acqs):
    print(i.file_path, index)
    i.calc_all_pdi(bands, cmr=False, map_channel=False, size=1000)

# %%
"""
Save data for kilosort
"""
for i in acqs:
    i.save_kilosort_bin(probe="acc")

# %%
"""
Load HDF5 data and KS data
"""
hdf5_path = r"D:\in_vivo_ephys\P16\acqs\2023_02_17\2023_02_17_M_L5_WT.hdf5"
ks_path = r"F:\acc_kilosort\2023_02_17_M_L5_WT_ivs"
acq_manager = AcqManager()
acq_manager.open_hdf5_file(hdf5_path)
acq_manager.load_kilosort(ks_path)

"""
Output data to use phy
"""
acq_manager.export_to_phy(
    ref_probe="acc",
    map_channel=True,
    probe="acc",
    chunk_size=480000,
    dtype="f32",
    output_chans=8,
)
