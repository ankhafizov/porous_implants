import numpy as np
import os
from file_paths import get_path
from PIL import Image
from pathlib import Path

def crop(img, shape, center=None):
    def left_edge(index):
        return np.ceil(center[index] - halves[index]).astype(np.int)

    def right_edge(index):
        return np.ceil(center[index] + halves[index] + odds[index]).astype(np.int)

    if center is None:
        center = [x // 2 for x in img.shape]

    halves = [x // 2 for x in shape]
    odds = [x % 2 for x in shape]

    ranges = (
        slice(
            left_edge(i),
            right_edge(i)
        )
        for i in range(img.ndim)
    )

    return img[tuple(ranges)]


def paste(matrix, fragmet, center):
    h, w = np.asarray(fragmet.shape)
    y0, x0 = np.asarray(center) - np.asarray(fragmet.shape) // 2
    matrix[y0:y0+h, x0:x0+w] = fragmet
    return matrix


def write_item_to_file(value, filename, db_folder):
    if not os.path.isdir(db_folder):
        os.mkdir(db_folder)

    file_path = os.path.join(db_folder, filename)
    file_txt = open(file_path,"a")
    file_txt.write(f"{value} \n")
    file_txt.close()


def get_2d_slice_of_sample_from_database(num_of_slice, file_id):
    data_folder = get_path(file_id)

    file_names = Path(data_folder).glob('*.tiff')
    file_names = list(file_names)
    img2d_gray = np.array(Image.open(file_names[num_of_slice]))

    return img2d_gray