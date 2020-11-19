from pathlib import Path
from PIL import Image
import numpy as np
import os
import h5py


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PHANTOM_DB_FOLDER_NAME = 'database'


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


data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
              '/2020_02/recon/123495/full_recon/20200206141126_123495/TiffSaver-tomo'

file_names = Path(data_folder).glob('*.tiff')
file_names = list(file_names)
print(len(file_names))

number_of_slices_in_group = 100
file_groups = split_list(file_names, number_of_slices_in_group)
print("total number of slice_groups: ", len(file_groups))

number_of_slice = 0
img3d = []
for file_name in file_groups[number_of_slice]:
    img2d = np.array(Image.open(file_name))
    img3d.append(img2d)
img3d=np.asarray(img3d)

print(img3d.shape)

save(img3d, 'test file.h5')
print(get_img('test file.h5').shape)