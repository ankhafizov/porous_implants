import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import file_paths
import data_manager as dm
from phase_contrast_restoration import get_img as get_bin_img
import os

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size':20})

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER_NAME = 'distributions'

def plot_hist(y, dist_type, sample_name):
    title = dist_type + ": " + sample_name

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(y, bins=100, edgecolor='k')

    ax.axvline(np.mean(y), color="red", label=f"mean {np.mean(y):.2f} mkm")
    ax.axvline(np.median(y), color="green", label=f"median {np.median(y):.2f} mkm")
    ax.legend()
    ax.set_title(f"{title} \n std: {np.std(y):.2f} mkm")
    ax.set_xlabel(f"{dist_type}, mkm")
    ax.set_ylabel(f"count")

    save_distribution(y, sample_name)
    return fig, np.mean(y), np.median(y), np.std(y)


def get_distribution(file_name):
    file_path = os.path.join(SCRIPT_PATH, DB_FOLDER_NAME, file_name)

    with h5py.File(file_path, 'r') as hdf:
        dataset = hdf.get(name = file_name)
        dataset = dataset[()]
    return dataset


def save_distribution(dist, file_name):
    db_folder = os.path.join(SCRIPT_PATH, DB_FOLDER_NAME)
    if not os.path.isdir(db_folder):
        os.mkdir(db_folder)

    db_path = os.path.join(db_folder, file_name)
    if os.path.isfile(db_path):
        os.remove(db_path)
    
    with h5py.File(db_path, 'a') as hdf:
        hdf.create_dataset(name = file_name, data = dist, compression='gzip', compression_opts=0)


if __name__=='__main__':
    df = dm.load_data("setup_culindric_diameters_and_distances.csv")
    # TODO: add dates
    # df = pd.DataFrame(columns = ['polimer_type',
    #                              'sample_number',
    #                              'median_diameters',
    #                              'mean_diameters',
    #                              'standard_dev_diameters',
    #                              'mean_distances',
    #                              'median_distances',
    #                              'standard_dev_distances']) 

    for polimer_type in ["PDL-05", "PDLG-5002"]:
        sample_id = 1

        paths = file_paths.get_benchtop_setup_paths(polimer_type)

        for sample_id in range(len(paths)):
            sample_name = list(paths.keys())[sample_id]

            data_info = sample_name.split()
            polimer, sample_number, date = data_info[0][:-2], data_info[0][-1], data_info[-1]
            img3d = get_bin_img(sample_name)

            snow = ps.networks.snow(im=img3d, voxel_size=9)
            proj = op.io.PoreSpy.import_data(snow)
            net, geom = proj

            pore_diameters, pore_distances = geom['pore.diameter'], geom['throat.length']

            title = f"DIAMETERS: {sample_name}"
            fig, mean_diameters, median_diameters, standard_dev_diameters = plot_hist(pore_diameters, 
                                                                                      "DIAMETERS",
                                                                                      sample_name)
            dm.save_plot(fig, "networks", f"{sample_id} " + title)

            title = f"DISTANCES: {sample_name}"
            fig, mean_distances, median_distances, standard_dev_distances = plot_hist(pore_distances,
                                                                                      "DISTANCES",
                                                                                      sample_name)
            dm.save_plot(fig, "networks", f"{sample_id} " + title)

            df = df.append({'polimer_type': polimer_type,
                            'sample_number': sample_number,
                            'median_diameters': median_diameters,
                            'mean_diameters': mean_diameters,
                            'standard_dev_diameters': standard_dev_diameters,
                            'mean_distances': mean_distances,
                            'median_distances': median_distances,
                            'standard_dev_distances': standard_dev_distances}, ignore_index=True)
            
            dm.save_dataframe(df, "setup_culindric_diameters_and_distances.csv")
            print("================", polimer_type, sample_id, " finished ================")