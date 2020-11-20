from pathlib import Path
from PIL import Image
import numpy as np
import os
import h5py
from scipy.signal import fftconvolve
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PHANTOM_DB_FOLDER_NAME = 'database'
TXT_FOULDER_NAME = 'txt_files'
TXT_FILE_NAME = 'k_values.txt'


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


def read_k_values(filename=TXT_FILE_NAME):
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
    f = interp1d(indexes_of_slices, k_values, kind='nearest')
    xnew = np.arange(0, max_number_of_slices, 1)
    return xnew, f(xnew)


if __name__=='__main__':
    data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                '/2020_02/recon/123495/full_recon/20200206141126_123495/TiffSaver-tomo'

    file_names = Path(data_folder).glob('*.tiff')
    file_names = list(file_names)
    N_fn = len(file_names)

    indexes_of_slices, k_values = read_k_values(filename=TXT_FILE_NAME)

    indexes_of_slices, k_values = interpolate_k_values(indexes_of_slices, k_values, N_fn)
    img3d_bin = []
    for (file_name, i, k) in zip(file_names, indexes_of_slices, k_values):
        img2d = np.array(Image.open(file_name))
        img2d_bin = binarize_slice(img2d, k=k, mu=25e-8)
        img3d_bin.append(img2d_bin)
        print(f'{i} out of {N_fn}')
    
    img3d_bin=np.asarray(img3d_bin)
    save(img3d_bin, f'bin_img.h5')
