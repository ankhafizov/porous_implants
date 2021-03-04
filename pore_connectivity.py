import porespy
import numpy as np

from phase_contrast_restoration import get_img as get_bin_img
from phase_contrast_restoration import save as save_to_h5
import file_paths

if __name__=='__main__':
    
    polimer_type = ["PDL-05", "PDLG-5002"][0]
    radius_coefs = {"PDL-05": 0.9, "PDLG-5002": 0.95}

    paths = file_paths.get_benchtop_setup_paths(polimer_type)

    for sample_id in range(len(paths)):
        sample_name = list(paths.keys())[sample_id]
        img3d = get_bin_img(sample_name)
        save_to_h5(~img3d, sample_name)