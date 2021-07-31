How to compile the code?
Run the main.py file and this project will auto run preprocess, data filter and CNN training.

The preprocess.py file will do preprocess about the IXI dataset including skull strip, N4_correction,
registration, normalisation, matter segment(GM,WM and CFS) and matter merge(GM+WM,GM+CFS and WM+CFS).
SimpleITK nibabel and numpy are required in this file.

The filter.py file will clear the data and make it correspond to the IXI.xls

The network_2_classes.py will do two way classification using different matters and combinations.

The close_age_range_classification.py will show the accuracy change in different age range with different matters


How to run and how to obtain the results?
The network_2_classes.py will generate 2 plots—— accuracy and loss

The close_age_range_classification.py will put all accuracy results in logfile.txt