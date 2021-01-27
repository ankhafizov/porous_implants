import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_closing, binary_fill_holes
from scipy.ndimage import convolve
from skimage import exposure
import closed_pores_evaluation as cpe
from file_paths import get_path
from pathlib import Path
from PIL import Image
import data_manager as dm
from helper import crop
from scipy.ndimage import label
from scipy.signal import fftconvolve

def get_2d_slice(num, file_id):
    data_folder = get_path(file_id)

    file_names = Path(data_folder).glob('*.tiff')
    file_names = list(file_names)
    img2d_gray = np.array(Image.open(file_names[num]))

    return crop(img2d_gray, (2000, 2000))


def filter_median(img, kernel_size = 3):
    kernel_shape = np.ones(img.ndim, dtype=int) * kernel_size
    kernel = np.ones(kernel_shape, dtype=int)
    # kernel = kernel / np.sum(kernel)
    return fftconvolve(img, kernel, mode='same')


file_id='123497'
num = 500
threshold = 0.9
max_contour_size = 10_000
img2d_gray = get_2d_slice(num, file_id)
#img2d_gray = filter_median(img2d_gray)

# img_equalized = exposure.equalize_hist(img2d_gray)
# img_equalized_binarized = img_equalized>0.5

min_brightness = np.min(img2d_gray)
max_brightness = np.max(img2d_gray)

img2d_gray = (img2d_gray - min_brightness) / (max_brightness - min_brightness)

prob_function = lambda x: np.exp(7*x**3)
probabilities = prob_function(np.linspace(0, 1, 100)) 
probabilities /= np.sum(probabilities) 

reference=np.random.choice(100, img2d_gray.shape, p=probabilities) / 100

img_matched = exposure.match_histograms(img2d_gray, reference, multichannel=True)

# Figure
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(15,30), constrained_layout=True)
axes = axes.flatten()
axes[0].imshow(img2d_gray, cmap="gray")
axes[1].imshow(img_matched, cmap="gray")

img_matched_bin = img_matched<threshold
_ = axes[2].hist(img_matched.flatten(), bins=255)
axes[2].axvline(x=threshold, color='red')
axes[3].imshow(img_matched_bin, cmap="gray")

mask = cpe.find_mask_longest_contours((img_matched>threshold),
                                      filter_by_contour_length=True,
                                      min_contour_length = max_contour_size,
                                      max_number_of_contours=None)

image_with_contours = binary_fill_holes(mask) 
axes[4].imshow(image_with_contours, cmap="gray")

imgl, _ = label(image_with_contours.astype(int))
axes[5].imshow(imgl, cmap="hsv")

#axes[6].imshow(mask, cmap="gray")
# axes[7].imshow(np.logical_and(image_with_contours, mask), cmap="gray")

for i, ax in enumerate(axes):
    if i!=2 and i!=4:
        ax.axis("off")

dm.save_plot(fig, 'hr_bin_plots', 'histogram transforms')

