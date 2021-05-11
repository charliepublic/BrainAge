import os
import subprocess


filePath = 'N4_Bias/IXI-T1/'
name_list = os.listdir(filePath)

outputPath = 'regist/IXI-T1/'
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
    cmd ="flirt -ref {} -in {}  -out {}".format(template_file, input_address, output_address)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    while True:
        line = p.stdout.readline()
        print(line.decode('gbk').strip("b'"))
        if line == b'' or subprocess.Popen.poll(p) == 0:
            p.stdout.close()
            break