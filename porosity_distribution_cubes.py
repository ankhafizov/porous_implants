from porespy.generators import blobs
import matplotlib.pyplot as plt
import numpy as np

import data_manager as dm
from helper import crop
from helper import crop, paste, get_2d_slice_of_sample_from_database
from phase_contrast_restoration import get_img as get_bin_img

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size':22})

PIXEL_SIZE_MM = 0.8125 * 10**-3

def divide_image_into_cubic_fragments(img, edge_size):
    count_of_center_points = np.min(img.shape) // edge_size
    img_fragments = []

    if img.ndim == 2:
        for x_coord in np.arange(count_of_center_points) + 0.5:
            for y_coord in np.arange(count_of_center_points) + 0.5:
                center_coords = np.ceil(np.asarray([x_coord, y_coord]) * edge_size).astype(int)
                img_fragment = crop(img, (edge_size, edge_size), center_coords)
                img_fragments.append(img_fragment)
    elif img.ndim == 3:
        for x_coord in np.arange(count_of_center_points) + 0.5:
            for y_coord in np.arange(count_of_center_points) + 0.5:
                for z_coord in np.arange(count_of_center_points) + 0.5:
                    center_coords = np.ceil(np.asarray([x_coord, y_coord, z_coord]) * edge_size).astype(int)
                    img_fragment = crop(img, (edge_size, edge_size, edge_size), center_coords)
                    img_fragments.append(img_fragment)

    return img_fragments


def get_porosity_histogram_disrtibution(img_fragments, file_id):
    get_porosity = lambda bin_img: np.sum(bin_img) / bin_img.size

    porosities = [get_porosity(img_fragment) for img_fragment in img_fragments]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(porosities, bins=25, edgecolor='k')
    ax.set_xlabel("porosity")
    ax.set_ylabel("count")
    ax.set_xlim([0,1])

    textstr = (f'$\sigma={np.std(porosities):.2f}$ \n $\mu={np.mean(porosities):.2f}$') + \
              (f'\n сторона кубика: {img_fragments[0].shape[0]*PIXEL_SIZE_MM} мм')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    ax.set_title(f"sample id {file_id}")

    return fig


if __name__=='__main__':
    file_id='123497'
    bin_img = ~get_bin_img(file_id+'.h5').astype(bool)

    img_fragments = divide_image_into_cubic_fragments(bin_img, edge_size=400)
    print(len(img_fragments))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(bin_img[0])
    dm.save_plot(fig, 'phantom section', 'section')

    dm.save_plot(get_porosity_histogram_disrtibution(img_fragments, file_id=file_id), 'phantom section', 'hist')
