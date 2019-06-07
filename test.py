import os

folder = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_train_img/"

valid_files = os.listdir(folder)

file_out = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_train_img/gfrc_train_img.txt"

file = open(file_out, 'w')

for ff in range(len(valid_files)):
    str_out = valid_files[ff] + '\n'
    file.write(str_out)

file.close()