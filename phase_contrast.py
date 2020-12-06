from pathlib import Path
from PIL import Image
import numpy as np
import os
import h5py
from scipy.signal import fftconvolve
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d
from helper import crop
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_closing
from skimage.morphology import disk, ball
from file_paths import get_path

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PHANTOM_DB_FOLDER_NAME = 'database'
TXT_FOULDER_NAME = 'txt_files'


def filter_mean(img):
    kernel_size = 3
    kernel_shape = np.ones(img.ndim, dtype=int) * kernel_size
    kernel = np.ones(kernel_shape, dtype=int)
    return fftconvolve(img, kernel, mode='same')


def split_list(lst, n):
    arr = []
    for i in range(0, len(lst), n):
        arr.append(lst[i:i + n])
    return arr


def save(img, file_name):
    db_folder = os.path.join(SCRIPT_PATH, PHANTOM_DB_FOLDER_NAME)
    if not os.path.isdir(db_folder):
        os.mkdir(db_folder)

    db_path = os.path.join(db_folder, file_name)
    if os.path.isfile(db_path):
        os.remove(db_path)
    
    with h5py.File(db_path, 'a') as hdf:
        hdf.create_dataset(name = file_name, data = img, compression='gzip', compression_opts=0)


def get_img(file_name):
    file_path = os.path.join(SCRIPT_PATH, PHANTOM_DB_FOLDER_NAME, file_name)

    with h5py.File(file_path, 'r') as hdf:
        dataset = hdf.get(name = file_name)
        dataset = dataset[()]
    return dataset


def poganins_correction(img, k=20, mu=25e-8):
    data_fft = np.fft.fft2(img)

    freq = np.fft.fftfreq(img.shape[0])
    fx, fy = np.meshgrid(freq, freq)
    f2 = np.sqrt(fx**2+fy**2)

    data_corr_fft = data_fft/np.sqrt((k * f2**2+mu))
    data_corr = np.abs(np.fft.ifft2(data_corr_fft))

    return data_corr


def binarize_slice(img2d, k=20, mu=25e-8):
    img2d_corrected = poganins_correction(img2d, k, mu)

    #probably look better
    img2d_corrected_filtered = filter_mean(img2d_corrected)
    thresh = threshold_otsu(img2d_corrected_filtered)

    return img2d_corrected_filtered > thresh


def binarize_volume(volume, k=20, mu=25e-8):
    volume_bin = []

    total_num_of_2d_slices = len(volume)
    for i, img2d in enumerate(volume):
        volume_bin.append(binarize_slice(img2d, k, mu))
        print(f"{i+1} out of {total_num_of_2d_slices}")

    return volume_bin


def read_k_values(filename):
    db_folder = os.path.join(SCRIPT_PATH, TXT_FOULDER_NAME, filename)
    file = open(db_folder, 'r')

    indexes_of_slices = []
    k_values = []
    for line in file:
        z, k = line.split()
        z, k = np.int(z), np.float32(k)
        k_values.append(k)
        indexes_of_slices.append(z)

    return indexes_of_slices, k_values


def interpolate_k_values(indexes_of_slices, k_values, max_number_of_slices):
    f = interp1d(indexes_of_slices, k_values, kind='nearest', fill_value="extrapolate")
    xnew = np.arange(0, max_number_of_slices, 1)
    return xnew, f(xnew)


def get_2d_mask(img2d, pad_width = 35, disk_radius=35, zoom_scale=0.1):
    merged_img = zoom(img2d, zoom_scale, order=1)
    result_paded = np.pad(merged_img,pad_width=((pad_width,pad_width),(pad_width,pad_width)), mode='constant')
    img_mask = binary_closing(result_paded, structure=disk(disk_radius))
    return crop(zoom(img_mask, 1/zoom_scale, order=1), img2d.shape)



def calculate_porosity_with_3d_mask(img3d, pad_width = 35, disk_radius=35, zoom_scale=0.1):
    # section_shape = img3d.shape[1:]
    # print('section_shape: ', section_shape)
    merged_img3d = zoom(img3d, zoom_scale, order=1)
    volume = 0
    body_volume = 0
    for img2d in merged_img3d:
        mask = get_2d_mask(img2d, pad_width = 35, disk_radius=35, zoom_scale=1)
        # mask = crop(zoom(mask, 1/zoom_scale, order=1), section_shape)
        volume += np.sum(mask)
        body_volume += np.sum(img2d)
    return 1 - body_volume / volume


# FILE_ID = '123493' good
# FILE_ID = '123494' good (mask needed)
# FILE_ID = '123495' good
# FILE_ID = '123496' good (mask needed)
# FILE_ID = '123497' good
# FILE_ID = '123498' good (mask needed)
# FILE_ID = '123499' good (mask needed)

if __name__=='__main__':
    # data_folder = get_path(FILE_ID)

    # file_names = Path(data_folder).glob('*.tiff')
    # file_names = list(file_names)
    # N_fn = len(file_names)

    # indexes_of_slices, k_values = read_k_values(filename=f'diamond {FILE_ID}.txt')

    # indexes_of_slices, k_values = interpolate_k_values(indexes_of_slices, k_values, N_fn)
    # img3d_bin = []
    # for (file_name, i, k) in zip(file_names, indexes_of_slices, k_values):
    #     img2d = np.array(Image.open(file_name))
    #     img2d_bin = binarize_slice(img2d, k=k, mu=25e-8)
    #     img3d_bin.append(img2d_bin)
    #     print(f'{i+1} out of {N_fn}')
    
    # img3d_bin=np.asarray(img3d_bin)
    # save(img3d_bin, f'{FILE_ID}.h5')
    # #print(f'porosity: {FILE_ID}', np.ones(img3d_bin)/img3d_bin.size)

    sample_params = [('123493', False),
                     ('123494', True),
                     ('123495', False),
                     ('123496', True),
                     ('123497', False),
                     ('123498', True),
                     ('123499', True)]

    for file_id, mask_needed in sample_params:
        img3d = get_img(f'{file_id}.h5')
        body_volume = np.sum(img3d)
        if mask_needed:
            print('mask_needed')
            porosity = calculate_porosity_with_3d_mask(img3d)
        else:
            print('mask NOT needed')
            sample_volume = img3d.shape[0] * img3d.shape[1] * img3d.shape[2]
            porosity = 1 - body_volume / sample_volume
        print(f'file_id: {file_id}, porosity: {porosity}')
        print("===============================")
