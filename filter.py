import os
import subprocess

import time

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
    time_now = time.strftime("%H", time.localtime())
    # if time_now == "23":
    #     os.system('shutdown -a')
# import pandas as pd
# #读取工作簿和工作簿中的工作表
# data_frame=pd.read_excel('IXI.xls')
# list = data_frame["IXI_ID"].tolist()
#
# i = 0
# for file_name in name_list:
#     i = i + 1
#     number = int(file_name.split("-")[0][3:])
#     if number not in list:
#         file_name = filePath + "/" + file_name
#         os.remove(file_name)
#
# print(i)
