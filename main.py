import subprocess

# pre-processing
cmd = "python extract.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python N4_Bias_Field_Correction.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python registration.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python normalisation.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python segement.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

# network
cmd = "python filter.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python network.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)