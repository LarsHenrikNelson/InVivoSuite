# %%
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from invivosuite.acq import (
    AcqManager,
)
from invivosuite.functions.spike_functions import get_template_channels


# %%
# Load both the hdf5 data.
hdf5_paths = sorted(list(Path("path/to/hdf5-files").rglob("*.hdf5")))
ks_paths = sorted(list(Path("path/to/kilosort-folders").glob("*")))
spk_acq_pair = []
for i in hdf5_paths:
    for j in ks_paths:
        if i.stem == j.stem:
            if "M" in i.stem.split("_"):
                spk_acq_pair.append((j, i))
spk_acq_pair = sorted(spk_acq_pair)


# %%
# This exports the average main waveform and template properties
save_path = Path(r"path/for/saving/data")
current_date = str(datetime.today().date())
for pair0, pair1 in spk_acq_pair:
    print(pair0.stem)

    acq_manager = AcqManager()
    acq_manager.load_hdf5(pair1)
    acq_manager.load_kilosort(pair0, load_type="r+")
    uid = acq_manager.file_path.stem
    out_path = Path(save_path) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting waveforms for {pair0.stem}")
    output = acq_manager.extract_waveforms(
        ref_probe="acc",
        map_channel=True,
        waveform_length=120,
        center=41,
        probe="acc",
        nchans=4,
        chunk_size=3000000,
        acq_type="spike",
        subtract=False,
    )
    print(f"Calculating templates for {pair0.stem}")
    filtered_templates, _ = acq_manager.extract_templates(
        spike_waveforms=output, total_chans=64
    )

    print(f"Calculating template properties for {pair0.stem}.")
    template_props = acq_manager.get_templates_properties(
        templates=filtered_templates,
        center=41,
        nchans=4,
        total_chans=64,
        upsample_factor=2,
    )

    _, channels = get_template_channels(filtered_templates, nchans=4, total_chans=64)

    # This gets the main channel of the filtered templates
    print(f"Extracting primary template for {pair0.stem}")
    j = np.arange(filtered_templates.shape[0])
    filtered_templates = filtered_templates[j, :, channels[:, 0]]

    print(f"Saving data for {pair0.stem}.")
    np.save(
        out_path / f"spike-templates-{current_date}.npy",
        filtered_templates,
    )

    template_df = pd.DataFrame(template_props)
    template_df.to_csv(
        out_path / f"template-properties-{current_date}.csv",
        index=False,
    )

    temp_ids = {}
    temp_ids = {
        "id": [uid] * acq_manager.cluster_ids.size,
        "cluster_id": acq_manager.cluster_ids,
    }

    temp_ids = pd.DataFrame(temp_ids)
    temp_ids.to_csv(out_path / f"template-ids-{current_date}.csv", index=False)

# %%
# This exports spike properties
save_path = Path(r"path/for/saving/data")
current_date = str(datetime.today().date())
for pair0, pair1 in spk_acq_pair:
    print(pair0.stem)

    acq_manager = AcqManager()
    acq_manager.load_hdf5(pair1)
    acq_manager.load_kilosort(pair0, load_type="r+")
    uid = acq_manager.file_path.stem
    out_path = Path(save_path) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    spk_props = acq_manager.get_spikes_properties(
        R=0.005,
        fs=40000.0,
        isi_threshold=0.0015,
        min_isi=0.0,
        nperseg=40,
    )

    spk_df = pd.DataFrame(spk_props)
    spk_df.to_csv(
        out_path / f"spike-properties-{current_date}.csv",
        index=False,
    )

# %%
# This exports spike burst properties. Burst finding uses the max-interval method.
save_path = Path(r"path/for/saving/data")
current_date = str(datetime.today().date())
for pair0, pair1 in spk_acq_pair:
    print(pair0.stem)

    acq_manager = AcqManager()
    acq_manager.load_hdf5(pair1)
    acq_manager.load_kilosort(pair0, load_type="r+")
    uid = acq_manager.file_path.stem
    out_path = Path(save_path) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    all_bursts_props, mean_bursts = acq_manager.get_burst_properties(
        min_count=5,
        min_dur=0.01,
        max_start=0.170,
        max_int=0.3,
        max_end=0.34,
        R=0.005,
        output_type="sec",
        fs=40000.0,
    )

    mean_df = pd.DataFrame(mean_bursts)
    mean_df.to_csv(
        out_path / f"bursts-properties-{current_date}.csv",
        index=False,
    )

    all_bursts_df = pd.DataFrame(all_bursts_props)
    all_bursts_df.to_csv(
        out_path / f"all-bursts-properties-{current_date}.csv",
        index=False,
    )

# %%
# This exports the STTC (synchrony) properties
save_path = Path(r"path/for/saving/data")
current_date = str(datetime.today().date())
dt = 5
for pair0, pair1 in spk_acq_pair:
    print(pair0.stem)

    acq_manager = AcqManager()
    acq_manager.load_hdf5(pair1)
    acq_manager.load_kilosort(pair0, load_type="r+")
    uid = acq_manager.file_path.stem
    out_path = Path(save_path) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    sttc_temp = acq_manager.compute_sttc(
        dt=dt, output_type="ms", fs=40000.0, sttc_version="ivs"
    )
    sttc_df = pd.DataFrame(sttc_temp)
    sttc_df.to_csv(
        out_path / f"sttc-{dt}ms-{current_date}.csv",
        index=False,
    )
