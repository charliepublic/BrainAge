import numpy as np
import os
import pandas as pd
import nibabel as nib
import scipy.ndimage.interpolation as Inter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'normalisation')

REFIST_DIR = os.path.join(DATA_DIR, 'IXI-T1')

filePath = 'normalisation/IXI-T1/'
name_list = os.listdir(filePath)
i = 0
file_list = []
for file_name in name_list:
    i = i + 1
    number = int(file_name.split("-")[0][3:])
    file_list.append(number)

print(len(file_list))
set1 = set(file_list)
table_path = os.path.join(ROOT_DIR, "IXI.xls")
df = pd.read_excel(table_path)
df = df.dropna(axis=0, how='any')
result = df.loc[:, ["IXI_ID", "AGE"]]

result = result[result["IXI_ID"].isin(file_list)]
result = result.drop_duplicates(subset=["IXI_ID"], keep='first')
print(len(result))

result.to_csv("new_IXI.csv", index=False)
id = result["IXI_ID"]
list_id = list(id)
delet_list = []

print("----------------")
for file_name in list_id:
    if file_name not in file_list:
        print(file_name)

print("----------------")
for file_name in file_list:
    if file_name not in list_id:
        delet_list.append(file_name)
        print(file_name)


for file_name in name_list:
    number = int(file_name.split("-")[0][3:])
    if number in delet_list:
        os.remove(filePath + file_name)

# resize img
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
filePath = 'normalisation/IXI-T1/'
name_list = os.listdir(filePath)
i = 0
file_list = []
outputPath = 'new_data/IXI-T1/'
try:
    os.makedirs(outputPath)
except:
    pass
for f in name_list:
    img = nib.load(os.path.join(REFIST_DIR, f))
    img_data = img.get_fdata()
    new_img_data = Inter.zoom(img_data, 0.5)
    new_image = nib.Nifti1Image(new_img_data, np.eye(4))
    i = i + 1
    print(f)
    output_address = outputPath + f
    nib.save(new_image, output_address)
