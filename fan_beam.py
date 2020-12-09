import porespy.generators as generator
import numpy as np
import astra_envelope as ae
import data_manager as dm
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

angles = np.arange(0, 180, 1)

blobns = 2
porsty = 0.5
shape = 100
phantom = generator.blobs([shape, shape], porosity=1 - porsty, blobiness=blobns)

source_object = 200
object_det = 200
sinogram_parallel = radon(phantom, theta=angles, circle=False)
sinogram_fan = np.transpose(ae.astra_fp_2d_fan(phantom,
                                               angles,
                                               source_object,
                                               object_det,
                                               detector_size=300))

fig, ax = plt.subplots(1,2)
ax[0].imshow(sinogram_fan, cmap='gray')
ax[0].set_title('sinogram_fan')
ax[1].imshow(sinogram_parallel, cmap='gray')
ax[1].set_title('sinogram_parallel')
dm.save_plot(fig, 'plots', 'sinograms')

print('--------------sinogram_fan-----------------')
column_nums=[np.random.randint(0, sinogram_parallel.shape[0]) for _ in range(10)]
for column_num in column_nums:
    print(f'integral of {column_num} of sinogram_fan:', np.sum(sinogram_fan[:, column_num]))
print(f'std: {np.std([np.sum(sinogram_fan[:, column_num]) for column_num in column_nums])}')

print('--------------sinogram_fan + ln[(R1+R2)/R1]-----------------')
sinogram_fan = sinogram_fan + np.log((source_object + object_det) / source_object)
for column_num in column_nums:
    print(f'integral of {column_num} of sinogram_fan:', np.sum(sinogram_fan[:, column_num]))
print(f'std: {np.std([np.sum(sinogram_fan[:, column_num]) for column_num in column_nums])}')

print('--------------sinogram_parallel-----------------')

for column_num in column_nums:
    print(f'integral of {column_num} of sinogram_parallel:', np.sum(sinogram_parallel[:, column_num]))
print(f'std: {np.std([np.sum(sinogram_parallel[:, column_num]) for column_num in column_nums])}')
