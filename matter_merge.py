import  nibabel as nib
import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_WM = os.path.join(ROOT_DIR, 'segment_WM')
REFIST_DIR_WM = os.path.join(DATA_DIR_WM, 'IXI-T1')

DATA_DIR_GM = os.path.join(ROOT_DIR, 'segment_GM')
REFIST_DIR_GM = os.path.join(DATA_DIR_GM, 'IXI-T1')

DATA_DIR_CFS = os.path.join(ROOT_DIR, 'segment_cfs')
REFIST_DIR_CFS = os.path.join(DATA_DIR_WM, 'IXI-T1')

file_list = os.listdir(REFIST_DIR_CFS)

REFIST_DIR_1 = REFIST_DIR_CFS
REFIST_DIR_2 = REFIST_DIR_GM
output_address = os.path.join(ROOT_DIR, 'segment_GM+CFS/IXI-T1/')
try:
    os.makedirs(output_address)
except:
    pass



for f in file_list:
    img_1 = nib.load(os.path.join(REFIST_DIR_1, f))
    img_data_1 = img_1.get_fdata()

    img_2 = nib.load(os.path.join(REFIST_DIR_2, f))
    img_data_2 = img_2.get_fdata()

    img_data_3 = img_data_1 + img_data_2

    output_path = output_address + f
    new_image = nib.Nifti1Image(img_data_3, np.eye(4))
    nib.save(new_image, output_path)
    print(f)