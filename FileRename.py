from fileinput import filename
import os

folder = r'/media/jerinpaul/New Volume/Datasets/Brain MRI KGL/Training/no_tumor/'
count = 0
for file_name in os.listdir(folder):
    source = folder + file_name
    destination = folder + "tst_img" + str(count) + ".jpg"
    count += 1
    os.rename(source, destination)
print('All Files Renamed')