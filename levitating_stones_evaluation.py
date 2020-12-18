import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from skimage.segmentation import clear_border

from data_manager import save_plot as save

VOXEL_SIZE = 9 * 1e-6

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


def get_closed_pores(bin_3d_volume, structure_neighbors_num=6):
    """
    function returns a bin_3d_volume of closed pores for the
    chosen neighbor voxels configuration

    """
    if not np.asarray(bin_3d_volume).dtype == bool:
        bin_3d_volume = bin_3d_volume.astype(bool)

    structure = get_structure(structure_neighbors_num)
    connected_components, _ = label(bin_3d_volume, structure)
    levitating_volume = clear_border(connected_components) > 0

    return levitating_volume


def recount_volumes_to_diameters(volumes):
    volumes = np.asarray(volumes)
    diameters = 2 * np.power((volumes*3)/(4*np.pi), 1/3)
    return diameters


def plot_pore_size_histogram(bin_3d_volume,
                             structure_neighbors_num=6,
                             size_type="volume",
                             num_of_bins=35,
                             max_x_value=None,
                             save_plot=False):
    """
    function returns a histogram of closed pores' sizes for the
    chosen neighbor voxels configuration.
    Note: size_type must be either "volume" or "diameter"
    """
    figure, ax = plt.subplots(figsize=(10, 10))

    lv = get_closed_pores(bin_3d_volume, structure_neighbors_num)
    total_num_of_pores = np.sum(lv)
    do_pores_exist = total_num_of_pores > 0

    if not do_pores_exist:
        ax.set_title('No pores detected', color="red")
        return None

    pore_size_distribution = get_pore_volume_distribution(lv,
                                                          structure_neighbors_num)
    if size_type == "volume":
        pass
    elif size_type == "diameter":
        pore_size_distribution = recount_volumes_to_diameters(pore_size_distribution)


    if not max_x_value:
        max_x_value = np.max(pore_size_distribution)

    bins = np.linspace(0, max_x_value, num_of_bins+1)

    stats = (f'total_num_of_pores = {total_num_of_pores} [voxels]\n'
             f'MAX pore {size_type} = {np.max(pore_size_distribution):.0f}\n'
             f'MIN pore {size_type} = {np.min(pore_size_distribution):.0f}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.35)
    ax.text(0.95, 0.3, stats, fontsize=9, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')

    ax.hist(pore_size_distribution,
            bins=bins,
            log=True,
            edgecolor='k')

    ax.set_title(f'pores_{size_type}_distribution | connectivity number={structure_neighbors_num}')
    ax.set_xlabel(f'{size_type} in voxels')
    ax.grid()

    if save_plot:
        save(figure, "plots", f"pores_{size_type}_distribution")


def get_pore_volume_distribution(levitatting_volume, structure_neighbors_num):
    connected_components, _ = label(levitatting_volume,
                                    get_structure(neighbors_num=structure_neighbors_num))
    pore_volume_distribution = np.unique(connected_components, return_counts=True)[1][1:]

    return pore_volume_distribution
