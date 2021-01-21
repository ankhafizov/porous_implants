import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as label_image
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


def filter_pores_mask(pore_mask_img,
                      lowest_value,
                      highest_value=None):
    labeled_img, _ = label_image(pore_mask_img)
    unique_labels, unique_counts = np.unique(labeled_img,
                                             return_counts=True)

    if highest_value == None:
        highest_value = np.max(unique_counts)+1
    accepted_labels = unique_labels[np.logical_and(unique_labels > 0,
                                                   unique_counts < highest_value,
                                                   unique_counts > lowest_value)]

    mask=np.zeros(pore_mask_img.shape)
    for elem in accepted_labels:
        # TODO: replace with create_mask_layer_for_label
        mask += np.where(elem == labeled_img, True, False)
    return mask


def create_mask_layer_for_label(labeled_img, label):
    mask = np.zeros(labeled_img.shape)
    mask +=  np.where(label == labeled_img, True, False)
    return mask.astype(bool)


def find_mask_longest_contours(bin_img_with_contours,
                               filter_by_contour_length=False,
                               max_number_of_contours=None,
                               min_contour_length=None):
    labeled_img, _ = label_image(bin_img_with_contours)
    unique_labels, unique_counts = np.unique(labeled_img,
                                             return_counts=True)
    
    longest_contours_indexes = np.flip(unique_counts.argsort())

    if not filter_by_contour_length:
        if len(longest_contours_indexes) < max_number_of_contours:
            max_number_of_contours = len(longest_contours_indexes)
        longest_contours_labels = unique_labels[longest_contours_indexes][0:max_number_of_contours]
    else:
        longest_contours_lens = unique_counts[longest_contours_indexes][0:max_number_of_contours]
        longest_contours_lens_lower_max = longest_contours_lens > min_contour_length
        longest_contours_indexes_sampled = longest_contours_indexes[longest_contours_lens_lower_max]
        longest_contours_labels = unique_labels[longest_contours_indexes_sampled]

    print("number of selected contour: ", len(longest_contours_labels))
    contour_mask = np.zeros(labeled_img.shape, dtype=bool)
    for label in longest_contours_labels:
        if label == 0:
            continue
        contour_mask = np.logical_or(contour_mask, create_mask_layer_for_label(labeled_img, label))
    return contour_mask


def hide_contours_in_image(img, contour_mask):
    contour_mask = contour_mask.astype(bool)
    return contour_mask*np.mean(img) + img * (~contour_mask)


def get_closed_pores(bin_3d_img,
                     structure_neighbors_num=6):
    """
    function returns a bin_3d_img of closed pores for the
    chosen neighbor voxels configuration

    """
    if not np.asarray(bin_3d_img).dtype == bool:
        bin_3d_img = bin_3d_img.astype(bool)

    structure = get_structure(structure_neighbors_num)
    connected_components, _ = label_image(bin_3d_img, structure)
    levitating_volume = clear_border(connected_components) > 0

    return levitating_volume


def recount_volumes_to_diameters(volumes, space_dim=2):
    volumes = np.asarray(volumes)
    if space_dim == 3:
        diameters = 2 * np.power((volumes*3)/(4*np.pi), 1/3)
    elif  space_dim == 2:
        diameters = 2 * np.power((volumes / np.pi), 1/2)
    else:
        raise ValueError("no such space. Should be \"2d\" or \"3d\"")
    return diameters


def plot_pore_size_histogram(closed_pores_mask,
                             structure_neighbors_num=6,
                             size_type="volume",
                             num_of_bins=35,
                             max_x_value=None,
                             min_x_value=0,
                             log_scale=True,
                             pixel_size_mkm=None,
                             add_median=False,
                             add_mean=False,
                             save_plot=False):
    """
    function returns a histogram of closed pores' sizes for the
    chosen neighbor voxels configuration.
    Note: size_type must be either "volume" or "diameter"
    """
    if pixel_size_mkm:
        unit_name="microns"
    else:
        unit_name="voxels"
        pixel_size_mkm=1

    figure, ax = plt.subplots(figsize=(10, 10))

    total_num_of_pores = np.sum(closed_pores_mask)
    do_pores_exist = total_num_of_pores > 0

    if not do_pores_exist:
        ax.set_title('No pores detected', color="red")
        return None

    pore_size_distribution = get_pore_volume_distribution(closed_pores_mask,
                                                          structure_neighbors_num)
    if size_type == "volume":
        pass
    elif size_type == "diameter":
        pore_size_distribution = recount_volumes_to_diameters(pore_size_distribution,
                                                              closed_pores_mask.ndim)
    
    pore_size_distribution = pore_size_distribution * pixel_size_mkm
    
    if not max_x_value:
        max_x_value = np.max(pore_size_distribution)
    if not min_x_value:
        min_x_value = np.min(pore_size_distribution)

    bins = np.linspace(min_x_value-1, max_x_value, num_of_bins+1)
    
    stats_median, stats_mean = "", ""
    if add_median:
        median_of_distribution = np.median(pore_size_distribution[pore_size_distribution > min_x_value])
        stats_median = f'Median {size_type} = {median_of_distribution:.2f} {unit_name}\n'
        ax.axvline(median_of_distribution, color="red", label="median")
    if add_mean:
        mean_of_distribution = np.mean(pore_size_distribution[pore_size_distribution > min_x_value])
        stats_mean = f'Mean {size_type} = {mean_of_distribution:.2f} {unit_name}'
        ax.axvline(mean_of_distribution, color="green", label="mean")

    stats = (f'MAX pore {size_type} = {max_x_value:.2f} {unit_name}\n'
             f'MIN pore {size_type} = {min_x_value:.2f} {unit_name}\n') + stats_median + stats_mean

    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.35)
    ax.text(0.95, 0.5, stats, fontsize=9, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='right')

    ax.hist(pore_size_distribution,
            bins=bins,
            log=log_scale,
            edgecolor='k')

    ax.set_title(f'pores {size_type} distribution | connectivity number={structure_neighbors_num}')
    ax.set_xlabel(f'{size_type} in {unit_name}')
    ax.set_ylabel('count')

    if add_mean or add_median:
        ax.legend()
    ax.grid()

    if save_plot:
        save(figure, "plots", f"pores_{size_type}_distribution")


def get_pore_volume_distribution(levitatting_volume, structure_neighbors_num):
    structure = get_structure(neighbors_num=structure_neighbors_num)
    if levitatting_volume.ndim == 2:
        structure = structure[1]
    connected_components, _ = label_image(levitatting_volume,
                                          structure)
    pore_volume_distribution = np.unique(connected_components, return_counts=True)[1][1:]

    return pore_volume_distribution
