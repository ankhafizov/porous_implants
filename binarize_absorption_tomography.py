import matplotlib.pyplot as plt
import numpy as np

import h5py
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter
from skimage.draw import disk

import file_paths
import data_manager as dm
import leviating_volume as lv
from phase_contrast_restoration import save as save_to_h5


plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size':22})
SAVE_IMG_DESKTOP_SETUP_FOLDER = 'setup sections preview'

# Вручную подобраны параметры для бинаризации polimer_attenuations_PDLG5002 и polimer_attenuations_PDL05

polimer_attenuations_PDLG5002 = ["low",
                                 "low",
                                 "high",
                                 "high",
                                 "low",
                                 "low", #manual 5
                                 "high", #manual 6
                                 "high", #manual 7
                                 "high",
                                 "high",
                                 "low",
                                 "high",
                                 "high",
                                 "high",
                                 "high"]

polimer_attenuations_PDL05 = ["low",
                              "low",
                              "low",
                              "low",
                              "low",
                              "low", #manual 5
                              "low", #manual 6
                              "low", #manual 7
                              "low", # 8
                              "high", # 9
                              "low", # 10
                              "low", # 11
                              "low", # 12
                              "low", # 13
                              "high"]


def binarize_without_eppendorf(img3d, polimer_attenuation):
    """
    polimer_attenuation = "low", "high"
    """
    img3d_bin = []
    for img2d in img3d:
        thresholds = threshold_multiotsu(img2d)
        if polimer_attenuation=="low":
            #img2d_bin = np.logical_and(img2d < thresholds[1], img2d > thresholds[0])
            # лучше вместе с эппендорфом
            img2d_bin = img2d > thresholds[0]
        elif polimer_attenuation=="high":
            img2d_bin = img2d > thresholds[1]
        
        img3d_bin.append(img2d_bin)
        
    return np.asarray(img3d_bin)


def plot_N_sections_multiotsu(img3d,
                              polimer_attenuation,
                              filename,
                              N=3,
                              radius_coef=0.9,
                              folder=SAVE_IMG_DESKTOP_SETUP_FOLDER):

    section_indexes = np.linspace(0, len(img3d), N, endpoint=False, dtype=int)
    fig, axes = plt.subplots(nrows=3, ncols=N, figsize=(21, 21))
    axes_plot, axes_hist, axes_bin = axes

    img3d = gaussian_filter(img3d, sigma=3)

    for ax_plot, ax_hist, i in zip(axes_plot, axes_hist, section_indexes):
        ax_plot.imshow(img3d[i], cmap="gray")

        ax_hist.hist(img3d[i].flatten(), bins=255, color="gray")
        thresholds = threshold_multiotsu(img3d[i])
        for thresh in thresholds:
            ax_hist.axvline(thresh, color='red')

    img3d = binarize_without_eppendorf(img3d, polimer_attenuation=polimer_attenuation)
    img3d = lv.remove_levitating_stones(img3d)

    for ax_bin, i in zip(axes_bin, section_indexes):
        ax_bin.imshow(img3d[i], cmap="gray", interpolation=None)
    
        center = np.asarray(img3d[i].shape) // 2

        rr, cc = disk(center, int(np.min(center)*radius_coef), shape=img3d[i].shape)
        mask = np.zeros(img3d[i].shape, dtype=int)
        mask[rr, cc] = True
        mask = np.ma.masked_where(mask>0, mask)
        ax_bin.imshow(mask, cmap="hsv", alpha=0.3)

    dm.save_plot(fig, folder, 'section '+filename)


if __name__=='__main__':
    # выбрать тип полимера
    polimer_type = ["PDL-05", "PDLG-5002"][0]
    sample_id = 10

    radius_coefs = {"PDL-05": 0.9, "PDLG-5002": 0.95}
    polimer_attenuation = {"PDL-05": polimer_attenuations_PDL05,
                           "PDLG-5002": polimer_attenuations_PDLG5002}
    
    paths = file_paths.get_benchtop_setup_paths(polimer_type)

    # for sample_id in [14]: #range(len(paths)):
    print(sample_id)
    sample_name = list(paths.keys())[sample_id]
    sample = h5py.File(paths[sample_name],'r')
    
    img3d = sample['Reconstruction'][:]
    
    # preview sections
    plot_N_sections_multiotsu(img3d,
                                polimer_attenuations_PDL05[sample_id],
                                str(sample_id) + ' ' + sample_name,
                                N=5)

    # binarize + save

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21, 14))
    for ax, i in zip(axes[0], [0, len(img3d) // 2, -1]):
        ax.imshow(img3d[i], cmap="gray")

    img3d = gaussian_filter(img3d, sigma=3)
    img3d = binarize_without_eppendorf(img3d, polimer_attenuation=polimer_attenuation[polimer_type][sample_id])
    img3d = lv.remove_levitating_stones(img3d)
    
    section_shape = img3d[0].shape
    center = np.asarray(section_shape) // 2
    rr, cc = disk(center, int(np.min(center)*radius_coefs[polimer_type]), shape=section_shape)
    mask = np.zeros(section_shape, dtype=int)
    mask[rr, cc] = True

    for i, img2d in enumerate(img3d):
        img3d[i] = mask * img2d

    for ax, i in zip(axes[1], [0, len(img3d) // 2, -1]):
        ax.imshow(img3d[i], cmap="gray")

    dm.save_plot(fig, "setup bin section", 'section '+ str(sample_id) + ' ' + sample_name)
    save_to_h5(img3d.astype(bool), sample_name)

    sample.close()