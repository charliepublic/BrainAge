import subprocess

# pre-processing
cmd = "python preprocess.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

# network
cmd = "python filter.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python network_segment_2_classes.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
cmd = "python close_age_range_classification.py"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)