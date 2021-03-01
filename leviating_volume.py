from scipy import ndimage
from skimage.segmentation import clear_border


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


def get_levitating_volume(volume, neighbors_num=6):
    """
    function returns a volume of levitating stones for the
    chosen neighbor voxels configuration
    """

    structure = get_structure(neighbors_num)
    connected_components, _ = ndimage.label(volume, structure)
    levitating_volume = clear_border(connected_components) > 0

    return levitating_volume


def remove_levitating_stones(image):
    levitating_stones = get_levitating_volume(
        image, neighbors_num=6
    )
    image[levitating_stones] = False

    return image