import numpy as np
from PIL import Image
from file_paths import get_path
from pathlib import Path
import matplotlib.pyplot as plt

import closed_pores_evaluation as cpe
import data_manager as dm


def get_2d_slice_of_sample_from_database(num_of_slice, file_id):
    data_folder = get_path(file_id)

    file_names = Path(data_folder).glob('*.tiff')
    file_names = list(file_names)
    img2d_gray = np.array(Image.open(file_names[num_of_slice]))

    return img2d_gray


def get_img2d_pores(img2d_grey,
                    percentile = 0.5,
                    min_pore_volume = 2,
                    max_pore_volume = 225):
    thresh_low = np.percentile(img2d_grey.ravel(), percentile)
    img2d_bin = img2d_grey < thresh_low
    img2d_pores = cpe.filter_pores_mask(img2d_bin,
                                        lowest_value = min_pore_volume,
                                        highest_value = max_pore_volume)
    return img2d_pores


def calc_histogram(img2d_pores,
                   pixel_size_mkm,
                   num_of_bins,
                   size_type,
                   min_x_value=2, 
                   max_x_value=None):
    pore_volume_distribution = cpe.get_pore_volume_distribution(img2d_pores, 6) * pixel_size_mkm
    if size_type == "diameter":
        print("max: ", np.max(pore_volume_distribution))
        pore_volume_distribution = cpe.recount_volumes_to_diameters(pore_volume_distribution,
                                                                    img2d_pores.ndim)
    elif size_type == "volume":
        pass
    else:
        raise ValueError(f"size_type has to be \"volume\" or \"diameter\", but \"{size_type}\" was given")

    if not max_x_value:
        max_x_value = np.max(pore_volume_distribution)
    bins = np.linspace(min_x_value, max_x_value, num_of_bins)
    return np.histogram(pore_volume_distribution, bins=bins)


def count_mean_from_hist(y, bins):
    bin_centers = (bins[1:] + bins[:-1])/2
    distr = []
    for y_elem, bin_center in zip(y, bin_centers):
        distr += [bin_center] * int(y_elem)
    return np.mean(distr)


def plot_error_hist(hist, err_hist, bins, ax, stats=None, bbox_fontsize=15):
    """
    stats = dict(unit_name="", size_type="", min_x_value, max_x_value ....)
    """
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    ax.bar(center,
           hist, 
           align='center',
           width=width,
           edgecolor ='k',
           yerr=err_hist,
           ecolor='r',
           error_kw=dict(capsize=3))
    ax.set_ylabel("count")

    if stats:
        ax.set_xlabel(stats["unit_name"])
        stats = (f'sample size: {stats["sample_size"]} layers\n'
                 f'total mean pore {stats["size_type"]} in layer:'
                 f' {stats["total_mean_pore_volume_in_layers"]:.2f} {stats["unit_name"]}\n'
                 f'MAX pore {stats["size_type"]} = {stats["max_x_value"]:.2f} {stats["unit_name"]}\n'
                 f'MIN pore {stats["size_type"]} = {stats["min_x_value"]:.2f} {stats["unit_name"]}\n')
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.35)
        ax.text(0.95,
                0.7,
                stats,
                fontsize=bbox_fontsize,
                bbox=bbox,
                transform=ax.transAxes, horizontalalignment='right')


if __name__=='__main__':
    file_ids=['123495', '12346', '123497', '123498']
    size_type = "diameter"
    size_of_sampling = 30

    for file_id in file_ids:
        percentile = 0.5
        min_pore_volume = 2
        max_pore_volume = 225
        pixel_size_mkm = 0.8125
        # size_type = "volume"

        fontsize=15
        plt.rcParams.update({'font.size':22})
        fig, ax = plt.subplots(figsize=(10,10))

        hists = []
        total_mean_pore_volume_in_layers = []
        for i in range(size_of_sampling):
            num = np.random.randint(0,2120)
            print("iter ", i+1, " out of ", size_of_sampling, " num: ", num)

            img2d_grey = get_2d_slice_of_sample_from_database(num, file_id=file_id)
            img2d_pores = get_img2d_pores(img2d_grey,
                                        percentile = percentile,
                                        min_pore_volume = min_pore_volume,
                                        max_pore_volume = max_pore_volume)
            total_mean_pore_volume_in_layers.append(np.sum(img2d_pores))
            hist, bins = calc_histogram(img2d_pores,
                                        pixel_size_mkm,
                                        size_type=size_type,
                                        num_of_bins=50)
            hists.append(hist)

        hist_mean = np.mean(hists, axis=0)
        hist_std = np.std(hists, axis=0) if size_of_sampling>1 else 0

        mean_of_distribution = count_mean_from_hist(hist_mean, bins)

        ax.axvline(mean_of_distribution, color="orange", label=f"mean={mean_of_distribution}")
        stats = dict(unit_name="mkm",
                    sample_size=size_of_sampling,
                    size_type=size_type,
                    min_x_value=min_pore_volume,
                    max_x_value=max_pore_volume,
                    total_mean_pore_volume_in_layers=np.mean(total_mean_pore_volume_in_layers) * pixel_size_mkm)
        plot_error_hist(hist_mean, hist_std, bins, ax, stats)
        ax.legend()
        ax.grid()
        dm.save_plot(fig, "plots", f"histogram {file_id} {size_type}")