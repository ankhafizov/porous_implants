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
sinogram_fan = np.transpose(ae.astra_fp_2d_fan(phantom, angles, 1000, 20))
sinogram_parallel = radon(phantom, theta=angles, circle=False)

fig, ax = plt.subplots(1,2)
ax[0].imshow(sinogram_fan, cmap='gray')
ax[0].set_title('sinogram_fan')
ax[1].imshow(sinogram_parallel, cmap='gray')
ax[1].set_title('sinogram_parallel')
dm.save_plot(fig, 'plots', 'sinograms')

column_nums=[10, 20, 40, 50, 130]
for column_num in column_nums:
    print(f'integral of {column_num} of sinogram_fan:', np.sum(sinogram_fan[:, column_num]))

print('-------------------------------')

for column_num in column_nums:
    print(f'integral of {column_num} of sinogram_parallel:', np.sum(sinogram_parallel[:, column_num]))