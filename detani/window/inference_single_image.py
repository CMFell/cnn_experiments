import os

import cv2
import numpy as np
import pandas as pd

from window.utils.tiling import create_tile_list
from window.models.window_classifier import AniWindowModel
from window.models.yolo_for_inference import YoloClass
from window.postprocess.match_truths import match_results_to_truths
from window.utils.drawing import draw_results_on_image
from window.utils.truths import windows_truth
from window.utils.inference_windows import process_annotation_df_negative_inference, create_windows_from_yolo, windows_to_whole_im


valid_whole_image_dir = "/data/old_home_dir/ChrissyF/GFRC/Valid/whole_images_all/"
truth_file = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid_GFRC_bboxes.csv"
image_file = "Z107_Img11169.jpg"

output_dir = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2_post/valid_results_on_image/"

# Process truth files
truths = pd.read_csv(truth_file)
truths['filename'] = [strin.replace('/', '_') for strin in truths.file_loc]
truths_im = truths[truths.filename == image_file]
truths_im = windows_truth(truths_im)

# Per image
whole_im = cv2.imread(valid_whole_image_dir + image_file)
whole_im = cv2.cvtColor(whole_im, cv2.COLOR_BGR2RGB)

# create tiles
tilez = create_tile_list(whole_im)

# create yolo model
saveweightspath = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2/testing_save_163.pt"
channels_in = 3
yolo_model = YoloClass(wtpath=saveweightspath, channels=channels_in)
boxes_whole_im = yolo_model.inference_on_image(tilez, 0.3)

# convert output positions to windows
windows_whole_im = process_annotation_df_negative_inference(boxes_whole_im, [1856, 1248])
windows_list = create_windows_from_yolo(windows_whole_im, tilez)

# classify windows
windows_classifier = AniWindowModel()
windows_filter_out = windows_classifier.inference_on_windows(windows_list, windows_whole_im)

# convert windows back to whole image
windows_results = windows_to_whole_im(windows_filter_out)

# calculate results
iou_threshold = 0.15
results_per_im = match_results_to_truths(windows_results, truths_im, iou_threshold)

# draw results on image
image_out = draw_results_on_image(whole_im, results_per_im)
cv2.imwrite(output_dir + image_file, image_out)

print('TP: ', np.sum(results_per_im.confmat == 'TP'), ' FP: ', np.sum(results_per_im.confmat == 'FP'), ' FN: ', np.sum(results_per_im.confmat == 'FN'))

