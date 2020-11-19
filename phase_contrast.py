from pathlib import Path
from PIL import Image
import numpy as np


def split_list(lst, n):
    arr = []
    for i in range(0, len(lst), n):
        print(i, i+n)
        arr.append(lst[i:i + n])
    return arr


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