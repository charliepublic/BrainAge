import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
print(len(set1))
table_path = os.path.join(ROOT_DIR, "IXI.xls")
df = pd.read_excel(table_path)
df = df.dropna(axis=0,how='any')
result = df.loc[:, ["IXI_ID", "AGE"]]
print(result)

result = result[result["IXI_ID"].isin(file_list)]
result = result.drop_duplicates(subset=["IXI_ID"], keep='first')
print(len(result))

result.to_csv("new_IXI.csv",index = False)
# id = result["IXI_ID"]
# list_id = list(id)
# print(list_id)
#
# for file_name in list_id:
#     if file_name not in file_list:
#         print(file_name)