import numpy as np
import data_manager as dm
import matplotlib.pyplot as plt
from icecream import ic
from skimage.filters import threshold_otsu

from helper import get_2d_slice_of_sample_from_database


# def poganins_correction(img, k=20, mu=25e-8, radius=0.001, n = 4):
#     data_fft = np.fft.fft2(img)

#     freq = np.sort(np.fft.fftfreq(img.shape[0]))
#     fx, fy = np.meshgrid(freq, freq)
#     f2 = np.sqrt(fx**2+fy**2)

#     # fourier_filter = lambda x: np.sqrt((k * x**2+mu))
#     # data_corr_fft = data_fft / fourier_filter(f2)

#     fourier_filter =  10 / (1 + 10 * (f2 / radius))
#     data_corr_fft = data_fft * fourier_filter

#     data_corr = np.abs(np.fft.ifft2(data_corr_fft))
#     ic(np.max(fourier_filter))
#     ic(np.min(fourier_filter))
#     ic(fourier_filter.shape)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.plot(fourier_filter[fourier_filter.shape[0] // 2])
#     ax.plot(f2[f2.shape[0] // 2])

#     dm.save_plot(fig, "unbiased_poganin", f"filter {file_id}, r=, n={n}")

#     return data_corr, fourier_filter


def poganins_correction(img, k=20, mu=25e-8):
    data_fft = np.fft.fft2(img)

    freq = np.fft.fftfreq(img.shape[0])
    fx, fy = np.meshgrid(freq, freq)
    f2 = np.sqrt(fx**2+fy**2)
    filter_func = 1 / np.sqrt(k * f2**2 + mu)

    data_corr_fft = data_fft * filter_func
    data_corr = np.abs(np.fft.ifft2(data_corr_fft))

    fourier_spectr_total_energy = np.sum(np.abs(data_fft)) ** 2
    fourier_spectr_filtered_energy = np.sum(np.abs(data_corr)) ** 2
    saved_energy = fourier_spectr_filtered_energy / fourier_spectr_total_energy

    return data_corr, saved_energy


def rescale_from_0_to_1(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


if __name__=='__main__':
    file_id = '123498'
    num = 1900

    img2d = get_2d_slice_of_sample_from_database(num, file_id)

    k_vals = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 15, 20]
    nrows = 3
    ncols = len(k_vals)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 7*nrows), constrained_layout=True)
    axes = np.transpose(axes)

    for row_index, (axes_row, k) in enumerate(zip(axes, k_vals)):

        img2d_restored, saved_energy = poganins_correction(img2d, k)
        thresh = threshold_otsu(img2d_restored) * 0.7

        if row_index == ncols // 2:
            axes_row[0].imshow(img2d, cmap='gray')
        else:
            axes_row[0].axis("off")
        
        axes_row[1].imshow(img2d_restored, cmap='gray')
        axes_row[1].set_title(f"saved energy: {saved_energy:.5f}, k: {k}")
        
        axes_row[2].imshow(img2d_restored > thresh, cmap='gray')
        axes_row[2].set_title(f"saved energy: {saved_energy:.5f}, k: {k}")

    dm.save_plot(fig, "unbiased_poganin", f"{file_id}")

    # k_vals = np.arange(0.1, 10, 0.5)
    # energies = []
    # ic(k_vals)
    # for k in k_vals:
    #     _, s = poganins_correction(img2d, k)
    #     energies.append(s)
    # 
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.plot(k_vals, energies)
    # ax.grid()
    # dm.save_plot(fig, "unbiased_poganin", f"{file_id}_energies")