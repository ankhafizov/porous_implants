import porespy
import numpy as np
import matplotlib.pyplot as plt
import h5py

from phase_contrast_restoration import get_img as get_bin_img
from phase_contrast_restoration import save as save_to_h5
import file_paths
import data_manager as dm

if __name__=='__main__':
    
    polimer_type = ["PDL-05", "PDLG-5002"][1]
    radius_coefs = {"PDL-05": 0.9, "PDLG-5002": 0.95}

    paths = file_paths.get_benchtop_setup_paths(polimer_type)

    for sample_id in range(len(paths)):
        sample_name = list(paths.keys())[sample_id]
        
        img3d_grey = h5py.File(paths[sample_name],'r')['Reconstruction'][:]
        img3d_bin = ~get_bin_img(sample_name)

        N = 3
        fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(21, 21))
        axes_grey, axes_bin = axes

        section_indexes = np.linspace(50, len(img3d_bin), N, endpoint=False, dtype=int)
        for ax_grey, ax_bin, i in zip(axes_grey, axes_bin, section_indexes):
            ax_grey.imshow(img3d_grey[i], cmap="gray")
            ax_bin.imshow(img3d_bin[i], cmap="gray")

        dm.save_plot(fig, "setup bin section", 'section '+ str(sample_id) + ' ' + sample_name)