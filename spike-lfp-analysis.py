# %%
from pathlib import Path
from datetime import datetime

import pandas as pd
from invivosuite.acq import AcqManager

# %%
hdf5_paths = sorted(list(Path("path/to/hdf5-files").rglob("*.hdf5")))
ks_paths = sorted(list(Path("path/to/kilosort-folders").glob("*")))
spk_acq_pair = []
for i in hdf5_paths:
    for j in ks_paths:
        if i.stem == j.stem:
            if "M" in i.stem.split("_"):
                spk_acq_pair.append((j, i))


# %%
save_path = Path(r"path/for/saving/data")
current_date = str(datetime.today().date())
for pair in spk_acq_pair[:1]:
    print(f"Starting {pair[0].stem} analysis")

    am = AcqManager()
    am.load_hdf5(pair[1])
    am.load_kilosort(pair[0])
    uid = am.file_path.stem
    out_path = Path(save_path) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    # Only need this if you have not set cwt settings.
    # am.set_cwt(
    #     f0=1,
    #     f1=110,
    #     fn=400,
    #     scaling="log",
    #     norm=True,
    #     nthreads=-1,
    # )

    all_phases, phase_stats = am.spike_phase(
        freq_bands={"gamma": [30, 80], "beta": [12, 20], "theta": [4, 10]},
        sxx_type="hilbert",
        ref_type="cmr",
        ref_probe="acc",
        map_channel=True,
        probe="acc",
    )

    # Option to use cwt instead of hilbert for spike-phase extraction
    # all_phases, phase_stats = am.spike_phase(
    #     freq_bands={"gamma": [30, 80], "beta": [12, 20], "theta": [4, 10]},
    #     sxx_type="cwt",
    #     ref_type="cmr",
    #     ref_probe="acc",
    #     map_channel=True,
    #     probe="acc",
    # )
    phase_stats_df = pd.DataFrame(phase_stats)
    phase_stats_df.to_csv(
        out_path / f"phase-properties-hilbert-{current_date}.csv", index=False
    )

    all_phases_df = pd.DataFrame(all_phases)
    all_phases_df.to_parquet(
        out_path / f"phases-hilbert-{current_date}.parquet.gzip", compression="gzip"
    )
