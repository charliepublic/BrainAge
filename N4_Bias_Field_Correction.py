import SimpleITK as sitk
import os


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


def main():
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


if __name__ == "__main__":
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    main()
