from porespy.generators import blobs
import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters.rank import mean as mean_filter
from skimage.morphology import ball

import data_manager as dm
from helper import crop
from helper import crop, paste, get_2d_slice_of_sample_from_database
from phase_contrast_restoration import get_img as get_bin_img
import file_paths
import leviating_volume as lv

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size':22})

PIXEL_SIZE_MM = 0.8125 * 10**-3
SAVE_IMG_SYNCHROTON_FOLDER = 'cubics synchrotron'
SAVE_IMG_DESKTOP_SETUP_FOLDER = 'cubics setup'


def divide_image_into_cubic_fragments(img, edge_size):
    count_of_center_points = np.asarray(img.shape) // edge_size
    img_fragments = []

    if img.ndim == 2:
        for x_coord in np.arange(count_of_center_points[0]+1) + 0.5:
            for y_coord in np.arange(count_of_center_points[1]+1) + 0.5:
                center_coords = np.ceil(np.asarray([x_coord, y_coord]) * edge_size).astype(int)
                img_fragment = crop(img, (edge_size, edge_size), center_coords)
                img_fragments.append(img_fragment)
    elif img.ndim == 3:
        for x_coord in np.arange(count_of_center_points[0]+1) + 0.5:
            for y_coord in np.arange(count_of_center_points[1]+1) + 0.5:
                for z_coord in np.arange(count_of_center_points[2]+1) + 0.5:
                    center_coords = np.ceil(np.asarray([x_coord, y_coord, z_coord]) * edge_size).astype(int)
                    img_fragment = crop(img, (edge_size, edge_size, edge_size), center_coords)
                    img_fragments.append(img_fragment)

    return img_fragments


def get_porosity_histogram_disrtibution(img_fragments, file_id, sample_shape, pixel_size_mm):
    get_porosity = lambda bin_img: (bin_img.size - np.sum(bin_img)) / bin_img.size

    porosities = [get_porosity(img_fragment) for img_fragment in img_fragments]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(porosities, bins=25, edgecolor='k')
    ax.set_xlabel("porosity")
    ax.set_ylabel("count")
    ax.set_xlim([0,1])

    textstr = (f'$\sigma={np.std(porosities):.2f}$;') + \
              (f'\n $\mu={np.mean(porosities):.2f}$;') + \
              (f'\n Размеры образца: \n {sample_shape[0]*pixel_size_mm:.2f}x') + \
              (f'{sample_shape[1]*pixel_size_mm:.2f}x{sample_shape[2]*pixel_size_mm:.2f} мм;') + \
              (f'\n Сторона кубика: {img_fragments[0].shape[0]*pixel_size_mm:.2f} мм;') + \
              (f'\n Кубиков: {len(img_fragments)} шт.')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    ax.set_title(f"sample id {file_id}")

    dm.save_plot(fig, SAVE_IMG_SYNCHROTON_FOLDER, f'hist {file_id}')


def plot_cubic_periodic_grid(img, ax, step):
    count_of_center_points = np.asarray(img.shape) // step
    print(count_of_center_points[0], count_of_center_points[1])

    for x_edge_coord in np.arange(count_of_center_points[0]+1)*step:
        for y_edge_coord in np.arange(count_of_center_points[1]+1)*step:
            
            print(x_edge_coord, y_edge_coord)
            ax.axhline(x_edge_coord, color='red', linewidth=2)
            ax.axvline(y_edge_coord, color='red', linewidth=2)

    return ax


def plot_edge_grid(ax, edges):
    (x1, x2), (y1, y2) = edges

    ax.axhline(y1, color='red', linewidth=4)
    ax.axhline(y2, color='red', linewidth=4)

    ax.axvline(x1, color='red', linewidth=4)
    ax.axvline(x2, color='red', linewidth=4)

    return ax


def save_first_section_of_img(img_3d, file_id, edge_size):
    with plt.style.context('classic'):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(img_3d[0], cmap='gray')
        ax = plot_cubic_periodic_grid(img_3d[0], ax, edge_size)
        ax.set_title(f'sample id {file_id}')
    dm.save_plot(fig, SAVE_IMG_SYNCHROTON_FOLDER, f'section {file_id}')


def calculate_synchrotron(edge_size = 400):
    # ============= Обработка синхротронных данных =====================
    # для неоцентрированных образцов
    sample_crop_edges = {'123493': (None, None),
                         '123494': (400, 1800),
                         '123495': (None, None),
                         '123496': (1200, 2100),
                         '123497': (None, None),
                         '123498': (1000, 1900)}
    file_id = '123498'
    # for file_id in sample_crop_edges.keys():
    # забираем уже отбинаризованый образец с папки database
    bin_img = get_bin_img(file_id+'.h5').astype(bool)

    # обрезаем картинку
    left_edge, right_edge = sample_crop_edges[file_id]
    if left_edge and right_edge:
        bin_img = bin_img[:, left_edge:right_edge, :] 

    # наризаем на кубики, показываем это на первом слое (сечении) и строим гистограмму
    img_fragments = divide_image_into_cubic_fragments(bin_img, edge_size=edge_size)
    save_first_section_of_img(bin_img, file_id, edge_size)

    get_porosity_histogram_disrtibution(img_fragments, file_id, bin_img.shape, PIXEL_SIZE_MM)


def plot_3_sections(img3d,
                    filename,
                    folder=SAVE_IMG_DESKTOP_SETUP_FOLDER,
                    grid_edges=None):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(21, 21))
    axes_plot, axes_hist, axes_bin = axes

    indexes = [0, len(img3d)//2, -1]

    img3d = gaussian_filter(img3d, sigma=3)
    thresh = threshold_otsu(img3d)

    for ax_plot, ax_hist, i in zip(axes_plot, axes_hist, indexes):
        ax_plot.imshow(img3d[i], cmap="gray")
        if grid_edges:
            plot_edge_grid(ax_plot, grid_edges)

        ax_hist.hist(img3d[i].flatten(), bins=255, color="gray")
        ax_hist.axvline(thresh, color='red')

    img3d = img3d > thresh
    img3d = lv.remove_levitating_stones(img3d)

    for ax_bin, i in zip(axes_bin, indexes):
        if grid_edges:
            plot_edge_grid(ax_bin, grid_edges)
        ax_bin.imshow(img3d[i], cmap="gray", interpolation=None)

    dm.save_plot(fig, folder, 'section '+filename)


# [(x1, x2), (y1, y2)]
sample_crop_edges_PDL05 = [[(100, 600), (100, 600)],
                           [(100, 600), (100, 600)],
                           [(100, 700), (100, 700)],
                           [(150, 775), (150, 775)],
                           [(100, 700), (100, 700)],
                           [(100, 600), (100, 600)],
                           [(100, 600), (100, 600)],
                           [(125, 675), (125, 675)],
                           [(150, 710), (100, 730)],
                           [(200, 800), (100, 800)],
                           [(120, 600), (120, 600)],
                           [(150, 650), (150, 650)],
                           [(200, 700), (200, 700)],
                           [(200, 750), (100, 800)],
                           [(150, 800), (200, 800)]]


sample_crop_edges_PDLG5002 = [[(100, 600), (100, 600)],
                              [(100, 600), (100, 600)],
                              [(100, 700), (100, 700)],
                              [(150, 775), (150, 775)],
                              [(100, 700), (100, 700)],
                              [(100, 600), (100, 600)],
                              [(100, 600), (100, 600)],
                              [(125, 675), (125, 675)],
                              [(150, 710), (100, 730)],
                              [(200, 800), (100, 800)],
                              [(120, 600), (120, 600)],
                              [(150, 650), (150, 650)],
                              [(200, 700), (200, 700)],
                              [(200, 750), (100, 800)],
                              [(150, 800), (200, 800)]]


if __name__=='__main__':
    polimer_type = ["PDL-05", "PDLG-5002"][1]
    paths = file_paths.get_benchtop_setup_paths(polimer_type)

    for sample_id in range(len(paths)):
        print(sample_id)
        sample_name = list(paths.keys())[sample_id]
        sample = h5py.File(paths[sample_name],'r')
        
        img3d = sample['Reconstruction'][:]

        grid_edges = sample_crop_edges_PDL05 if polimer_type=="PDL-05" else sample_crop_edges_PDLG5002
        plot_3_sections(img3d, 
                        str(sample_id) + ' ' + sample_name,
                        grid_edges=sample_crop_edges_PDL05[sample_id])

    sample.close()



    
