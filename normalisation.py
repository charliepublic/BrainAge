import os
import nibabel as nib
from sklearn.mixture import GaussianMixture

from intensity_normalization.normalize import gmm

filePath = 'regist/IXI-T1/'
name_list = os.listdir(filePath)
outputPath = 'normalisation/IXI-T1/'
try:
    processed_list = os.listdir(outputPath)
except:
    os.makedirs(outputPath)
    processed_list = os.listdir(outputPath)

for file_name in name_list:
    print(file_name)
    if file_name in processed_list:
        continue
    input_address = filePath + file_name
    output_address = outputPath + file_name
    try:
        proxy = nib.load(input_address, keep_file_open=False)
        normalised = gmm.gmm_normalize(proxy)
        nib.save(normalised, output_address)
    except:
        print('！！！！error！！！！！！')
        print(file_name)
        print('!!!!!!!!!!!!!!!!!!')
        continue