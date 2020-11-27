import numpy as np

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