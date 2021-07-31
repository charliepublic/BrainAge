import os
import subprocess

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from intensity_normalization.normalize import gmm


def skll_strip():
    # skull strip
    filePath = 'IXI-T1/'
    name_list = os.listdir(filePath)

    stripPath = 'Strip/'

    processd_list = os.listdir("Strip/IXI-T1/")

    for file_name in name_list:
        print(file_name)
        if file_name in processd_list:
            continue
        file_name = filePath + file_name
        file_address = stripPath + file_name
        cmd = "D:\Code_Libaray\Python\project\ROBEX\\runROBEX.bat " + file_name + " " + file_address
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        while True:
            line = p.stdout.readline()
            print(line.decode('gbk').strip("b'"))
            if line == b'' or subprocess.Popen.poll(p) == 0:
                p.stdout.close()
                break


def correct(inputImage, maskImage):
    output = corrector.Execute(inputImage, maskImage)
    return output


def N4(input_path, output_path):
    print("N4 bias correction runs.")
    inputImage = sitk.ReadImage(input_path)
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    sitk.WriteImage(maskImage, "06-t1c_mask3.nii.gz")
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    output = correct(inputImage, maskImage)
    sitk.WriteImage(output, output_path)
    print("Finished N4 Bias Field Correction.....")


def N4_correction():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filePath = 'Strip/IXI-T1/'
    name_list = os.listdir(filePath)

    processPath = 'N4_Bias/IXI-T1/'

    try:
        processed_list = os.listdir(processPath)
    except:
        os.makedirs(processPath)
        processed_list = os.listdir(processPath)

    for file_name in name_list:
        print(file_name)
        if file_name in processed_list:
            continue

        input_path = filePath + file_name
        output_path = processPath + file_name

        N4(input_path, output_path)


def registration():
    filePath = 'N4_Bias/IXI-T1/'
    name_list = os.listdir(filePath)

    outputPath = 'registration/IXI-T1/'
    try:
        processd_list = os.listdir(outputPath)
    except:
        os.makedirs(outputPath)

    for file_name in name_list:
        print(file_name)
        if file_name in processd_list:
            continue
        input_address = filePath + file_name
        output_address = outputPath + file_name

        template_file = "MNI152_T1_1mm_brain.nii.gz"
        cmd = "flirt -ref {} -in {}  -out {}".format(template_file, input_address, output_address)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        while True:
            line = p.stdout.readline()
            print(line.decode('gbk').strip("b'"))
            if line == b'' or subprocess.Popen.poll(p) == 0:
                p.stdout.close()
                break


def normalisation():
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


def segment():
    filePath = 'normalisation/IXI-T1/'
    name_list = os.listdir(filePath)

    outputPath = 'segment/IXI-T1/'
    try:
        processd_list = os.listdir(outputPath)
        print(processd_list)
    except:
        os.makedirs(outputPath)
        processd_list = os.listdir(outputPath)

    for file_name in name_list:
        print(file_name)
        if file_name in processd_list:
            continue
        input_address = filePath + file_name
        output_address = outputPath + file_name + "/"
        os.makedirs(output_address)
        template_file = "MNI152_T1_1mm_brain.nii.gz"
        cmd = "fast -t 1 -n 3 -o {} {}".format(output_address, input_address)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        while True:
            line = p.stdout.readline()
            print(line.decode('gbk').strip("b'"))
            if line == b'' or subprocess.Popen.poll(p) == 0:
                p.stdout.close()
                break


def matter_merge():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR_WM = os.path.join(ROOT_DIR, 'segment_WM')
    REFIST_DIR_WM = os.path.join(DATA_DIR_WM, 'IXI-T1')

    DATA_DIR_GM = os.path.join(ROOT_DIR, 'segment_GM')
    REFIST_DIR_GM = os.path.join(DATA_DIR_GM, 'IXI-T1')

    DATA_DIR_CFS = os.path.join(ROOT_DIR, 'segment_cfs')
    REFIST_DIR_CFS = os.path.join(DATA_DIR_WM, 'IXI-T1')

    file_list = os.listdir(REFIST_DIR_CFS)

    # this only generate GM and CFS
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


if __name__ == "__main__":
    skll_strip()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    N4_correction()
    registration()
    normalisation()
    segment()
    matter_merge()
