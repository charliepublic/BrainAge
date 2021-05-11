import os
import subprocess


filePath = 'normalisation/IXI-T1/'
name_list = os.listdir(filePath)

outputPath = 'segement/IXI-T1/'
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
    output_address = outputPath + file_name +"/"
    os.makedirs(output_address)
    template_file = "MNI152_T1_1mm_brain.nii.gz"
    cmd ="fast -t 1 -n 3 -o {} {}".format(output_address, input_address)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    while True:
        line = p.stdout.readline()
        print(line.decode('gbk').strip("b'"))
        if line == b'' or subprocess.Popen.poll(p) == 0:
            p.stdout.close()
            break