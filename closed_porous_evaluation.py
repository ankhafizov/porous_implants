import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.segmentation import clear_border

import data_manager as dm


def get_structure(neighbors_num=6):
    """
    function for determine which voxels we consider as neighbor ones
    :type neighbors_num: int, shud be 6,18 or 26
    """
    if neighbors_num == 6:
        structure = [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]

    elif neighbors_num == 18:
        structure = [
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        ]

    elif neighbors_num == 26:
        structure = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        ]
    else:
        raise ValueError('You should choose neighbors_num from  [6, 18, 26]')

    return structure


def get_closed_pores(material_volume, neighbors_num=6):
    """
    function returns a volume of closed pores distribution 
    for the chosen neighbor voxels configuration.
    material_volume denotes material distribution of 
    the porous sample (i.e. its frame)
    """
    volume =  ~material_volume.astype(bool)

    structure = get_structure(neighbors_num)
    connected_components, _ = ndimage.label(volume, structure)
    closed_pores_volume = clear_border(connected_components) > 0

    return closed_pores_volume


def image_3d_preview(folder, image, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(image.astype(int), edgecolor='k')
    plt.title(title or '3d preview')
    title = title + '.jpg'
    dm.save_plot(figure=fig, folder_name=folder, name=title)



def plot_pore_volume_hist(material_volume,
                          structure_neighbors_num=6,
                          num_of_bins=35,
                          max_x_value=350,
                          log_scale=False,
                          ax=None):

    cp = get_closed_pores(material_volume, structure_neighbors_num)
    total_num_of_pores = np.sum(cp)
    print('total_num_of_pores ',total_num_of_pores)
    do_pores_exist = total_num_of_pores > 0

    if not do_pores_exist:
        ax.set_title('No pores - no plots')
        return []

    volume_distribution = get_pore_volume_distribution(cp,
                                                       structure_neighbors_num)
    bins = np.linspace(0, max_x_value, num_of_bins+1)

    stats = (f'total_num_of_pores = {total_num_of_pores}\n'
             f'max_volume = {np.max(volume_distribution):.0f}\n'
             f'min_volume = {np.min(volume_distribution):.0f}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.35)
    ax.text(0.95, 0.3, stats, fontsize=9, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')

    ax.hist(volume_distribution,
            bins=bins,
            log=log_scale,
            edgecolor='k',
            label='all pores')

    ax.set_title(f'pores_volume_distribution | structure_neighbors_num={structure_neighbors_num}')

    ax.set_xlabel('volume in voxels')
    ax.legend()
    ax.grid()


def get_pore_volume_distribution(closed_pores_volume, structure_neighbors_num):
    connected_components, _ = ndimage.label(closed_pores_volume,
                                            get_structure(neighbors_num=structure_neighbors_num))
    pore_volume_distribution = np.unique(connected_components, return_counts=True)[1][1:]

    return pore_volume_distribution


def get_large_pore_coords(material_volume, structure_neighbors_num):

    cp = get_closed_pores(material_volume, structure_neighbors_num)
    connected_components, _ = ndimage.label(cp,
                                            get_structure(neighbors_num=structure_neighbors_num))
    unique_elements = np.unique(connected_components, return_counts=True)
    pore_volume_distribution = unique_elements[1][1:]
    pore_volume_labels = unique_elements[0][1:]

    large_pore = np.sort(pore_volume_distribution)[-1]
    index_of_large_pore = np.where(pore_volume_distribution == large_pore)
    label_of_large_pore = pore_volume_labels[index_of_large_pore][0]

    all_coords = np.where(connected_components == label_of_large_pore)
    center_coords = []
    for i in range(len(all_coords)):
        center_coords.append(int(np.mean(all_coords[i])))
    center_coords = np.asarray(center_coords)

    return center_coords


