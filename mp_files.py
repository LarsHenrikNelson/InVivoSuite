# from multiprocessing import Pool
from pathlib import Path

from joblib import Parallel, delayed

from invivosuite.acq import AcqManager


def main():
    pc_paths = {
        # "L4": "D:/in_vivo_ephys/2022_12_16/L4",
        # "R4": "D:/in_vivo_ephys/2022_12_14/R4",
        # "R5": "D:/in_vivo_ephys/2022_12_14/R5",
        # "L5": "D:/in_vivo_ephys/2022_12_14/L5",
        "FN_WT": "D:/in_vivo_ephys/2023_02_17/FN_WT",
        "FKO": "D:/in_vivo_ephys/2023_02_17/FL5_KO",
        "MWT": "D:/in_vivo_ephys/2023_02_17/ML5_WT",
        "ML4": "D:/in_vivo_ephys/2023_02_24/ML4",
        "ML5": "D:/in_vivo_ephys/2023_02_24/ML5",
        "ML1_WT": "D:/in_vivo_ephys/2023_03_03/ML1_WT",
        "ML4_KO": "D:/in_vivo_ephys/2023_03_03/ML4_KO",
        "MN_KO": "D:/in_vivo_ephys/2023_03_08/MN_KO_P16",
        "FL_WT": "D:/in_vivo_ephys/2023_03_09/FL",
        # "ML_WT": "D:/in_vivo_ephys/2023_03_09/ML",
        "MLL_WT": "D:/in_vivo_ephys/2023_03_09/MLL",
        # "FL5_KO": "D:/in_vivo_ephys/2023_03_17/FL5_KO",
        "FWT": "D:/in_vivo_ephys/2023_03_20/FWT",
    }
    mac_paths = {
        # "L4": "/Volumes/Backup/in_vivo_ephys/2022_12_16/L4",
        # "R4": "/Volumes/Backup/in_vivo_ephys/2022_12_14/R4",
        # "R5": "/Volumes/Backup/in_vivo_ephys/2022_12_14/R5",
        # "L5": "/Volumes/Backup/in_vivo_ephys/2022_12_14/L5",
        "FN_WT": "/Volumes/Backup/in_vivo_ephys/2023_02_17/FN_WT",
        "FKO": "/Volumes/Backup/in_vivo_ephys/2023_02_17/FL5_KO",
        "MWT": "/Volumes/Backup/in_vivo_ephys/2023_02_17/ML5_WT",
        "ML4": "/Volumes/Backup/in_vivo_ephys/2023_02_24/ML4",
        "ML5": "/Volumes/Backup/in_vivo_ephys/2023_02_24/ML5",
        "ML1_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_03/ML1_WT",
        "ML4_KO": "/Volumes/Backup/in_vivo_ephys/2023_03_03/ML4_KO",
        "MN_KO": "/Volumes/Backup/in_vivo_ephys/2023_03_08/MN_KO_P16",
        "FL_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_09/FL",
        "ML_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_09/ML",
        "MLL_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_09/MLL",
        "FL5_KO": "/Volumes/Backup/in_vivo_ephys/2023_03_17/FL5_KO",
        "FWT": "/Volumes/Backup/in_vivo_ephys/2023_03_20/FWT",
    }
    for i in pc_paths.values():
        m = Path(i)
        paths = list(m.rglob("*.hdf5"))
        # with Pool(2) as p:
        #     results = p.imap_unordered(func=process_object, iterable=paths)
        #     for i in results:
        #         print("Started new job")
        #         i
        #     p.close()
        #     p.join()

        Parallel(n_jobs=4)(delayed(process_object)(x) for x in paths)


def process_object(path):
    obj = AcqManager()
    obj.load_hdf5_file(path)
    obj.find_lfp_bursts(
        window="hamming",
        min_len=0.2,
        max_len=20,
        min_burst_int=0.2,
        wlen=200,
        threshold=10,
        pre=3,
        post=3,
        order=100,
        method="spline",
        tol=0.001,
        deg=90,
    )
    obj.close()


if __name__ == "__main__":
    main()
