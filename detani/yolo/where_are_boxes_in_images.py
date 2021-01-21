import pandas as pd
import cv2
import numpy as np

"""
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_img/'
annot_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train768_ann/'
out_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/check_input/'
train_file = base_dir + "gfrc_train.txt"

input_file = pd.read_csv(train_file)
test_gt = input_file['gt_details']
images_in = input_file['img_name']

for ff in range(10,30):
    read_img = cv2.imread(base_dir + images_in[ff])
    rez_path = annot_dir + test_gt[ff]
    rez_gt = pd.read_csv(rez_path, sep=' ', header=None)
    xc = rez_gt[rez_gt.columns[1]]
    yc = rez_gt[rez_gt.columns[2]]
    xw = rez_gt[rez_gt.columns[3]]
    yh = rez_gt[rez_gt.columns[4]]
    xmn = np.multiply(np.subtract(xc, np.divide(xw,2)), 1152)
    xmx = np.multiply(np.add(xc, np.divide(xw,2)), 1152)
    ymn = np.multiply(np.subtract(yc, np.divide(yh, 2)), 768)
    ymx = np.multiply(np.add(yc, np.divide(yh, 2)), 768)
    for rw in range(rez_gt.shape[0]):
        cv2.rectangle(read_img, (int(xmn[rw]), int(ymn[rw])), (int(xmx[rw]), int(ymx[rw])), (255, 0, 0), 2)
    cv2.imwrite(out_dir + images_in[ff], read_img)
"""

base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_valid_img/'
tp_file= 'E:/CF_Calcs/BenchmarkSets/GFRC/copy_out/darknet384576_5pcblanks/tp.csv'
fp_file= 'E:/CF_Calcs/BenchmarkSets/GFRC/copy_out/darknet384576_5pcblanks/tp.csv'
fn_file= 'E:/CF_Calcs/BenchmarkSets/GFRC/copy_out/darknet384576_5pcblanks/tp.csv'
out_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/check_input/'

tptp = pd.read_csv(tp_file)
tptp["type_res"] = np.repeat("tp", tptp.shape[0])
fpfp = pd.read_csv(fp_file)
fpfp["type_res"] = np.repeat("fp", fpfp.shape[0])
fnfn = pd.read_csv(fn_file)
fnfn["type_res"] = np.repeat("fn", fpfp.shape[0])

frames = [tptp, fpfp, fnfn]

results = pd.concat(frames)

pics = np.unique(results.file)

for ff in range(30):
    read_img = cv2.imread(base_dir + pics[ff] + '.png')
    rez_mask = results.file == pics[ff]
    rez_gt = results[rez_mask]
    xmn = rez_gt.xmin.tolist()
    xmx = rez_gt.xmax.tolist()
    ymn = rez_gt.ymin.tolist()
    ymx = rez_gt.ymax.tolist()
    for rw in range(rez_gt.shape[0]):
        cv2.rectangle(read_img, (int(xmn[rw]), int(ymn[rw])), (int(xmx[rw]), int(ymx[rw])), (255, 0, 0), 2)
    print(out_dir + pics[ff] + '.png')
    cv2.imwrite(out_dir + pics[ff] + '.png', read_img)





