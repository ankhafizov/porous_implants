from porespy import visualization
from phase_contrast_restoration import get_img as get_bin_img
import file_paths
import matplotlib.pyplot as plt
import data_manager as dm

from skimage.measure import marching_cubes

polimer_type = ["PDL-05", "PDLG-5002"][0]
sample_ids = {"PDL-05": [10, 12, 14],  "PDLG-5002": [0, 2, 4]}
for sample_id in sample_ids[polimer_type]:
    paths = file_paths.get_benchtop_setup_paths(polimer_type)
    sample_name = list(paths.keys())[sample_id]

    img3d = get_bin_img(sample_name)
    img2d = visualization.show_3D(img3d)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img2d)
    dm.save_plot(fig, "3d visualization", f'{sample_id} 3d ' + sample_name)
    print("ready: ", sample_id)

    # visualization.show_mesh(marching_cubes())