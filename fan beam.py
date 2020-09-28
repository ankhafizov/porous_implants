import porespy.generators as generator
import numpy as np
import astra_envelope as ae

angles = np.arange(0, 180, 1)

blobns = 2
porsty = 0.5
shape = 100
phantom = generator.blobs([shape, shape], porosity=1 - porsty, blobiness=blobns)
sinogram = ae.astra_fp_2d_fan(phantom, angles, 100, 20)
print(sinogram)