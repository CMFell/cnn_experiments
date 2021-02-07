import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms as transforms
import torchvision.transforms.functional as tf

from window.utils.tiling import create_tile_list
from window.models.train_model import AniWindowModel
from window.models.window_classifier import BinaryVggClassifier, BinaryWindowClassifier
from window.models.yolo_for_inference import YoloClass
from window.postprocess.match_truths import match_results_to_truths
from window.utils.drawing import draw_results_on_image
from window.utils.truths import windows_truth
from window.utils.inference_windows import process_annotation_df_negative_inference, create_windows_from_yolo, windows_to_whole_im

output_base_dir = "/home/cmf21/pytorch_save/GFRC/Bin/post_processed/pos_from_yolo/rgb_baseline2_incep/"

valid_whole_image_dir = "/data/old_home_dir/ChrissyF/GFRC/Valid/whole_images_all/"
truth_file = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid_GFRC_bboxes.csv"

output_dir = output_base_dir + "valid_results_on_image/"
output_csv = output_base_dir + "results_post_window_classifier.csv"

# get all image files
image_files = list(Path(valid_whole_image_dir).glob("*.jpg"))
image_files = [img.name for img in image_files]

# Process truth files
truths = pd.read_csv(truth_file)
truths.loc[:, 'filename'] = [strin.replace('/', '_') for strin in truths.file_loc]

# create yolo model
saveweightspath = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2/testing_save_163.pt"
channels_in = 3
yolo_model = YoloClass(wtpath=saveweightspath, channels=channels_in)

# create window classifier
cp_path = output_base_dir + "patch_model/checkpoint.ckpt.ckpt"
classifier = BinaryWindowClassifier.load_from_checkpoint(checkpoint_path=cp_path)
windows_classifier = AniWindowModel(classifier)

# create empty list to store results
results_all_ims = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal', 'confmat', 'tru_box', 'filename'])

for fl in image_files:
    # filter truths
    truths_im = truths[truths.filename == fl]
    truths_im_pixels = windows_truth(truths_im)
    
    # Per image
    print(fl)
    whole_im = cv2.imread(valid_whole_image_dir + fl)
    whole_im = cv2.cvtColor(whole_im, cv2.COLOR_BGR2RGB)
    
    # create tiles
    tilez = create_tile_list(whole_im)

    # get predictions from yolo
    boxes_whole_im = yolo_model.inference_on_image(tilez, 0.3)

    if boxes_whole_im.shape[0] > 0:
        # convert output positions to windows
        windows_whole_im = process_annotation_df_negative_inference(boxes_whole_im, [1856, 1248])
        windows_list = create_windows_from_yolo(windows_whole_im, tilez)

        # classify windows
        windows_filter_out = windows_classifier.inference_on_windows(windows_list, windows_whole_im)

        # convert windows back to whole image
        windows_results = windows_to_whole_im(windows_filter_out)
    
    else:
        windows_results = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])

    # calculate results
    iou_threshold = 0.15
    nms_threshold = 0.1
    results_per_im = match_results_to_truths(windows_results, truths_im_pixels, iou_threshold, nms_threshold)
    results_per_im['filename'] = fl

    results_all_ims = pd.concat((results_all_ims, results_per_im), axis=0, sort=False)

    # draw results on image
    image_out = draw_results_on_image(whole_im, results_per_im)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_dir + fl, image_out)

results_all_ims.to_csv(output_csv, index=False)

print(f'TP: {np.sum(results_all_ims.confmat=="TP")}, FP: {np.sum(results_all_ims.confmat=="FP")}, FN: {np.sum(results_all_ims.confmat=="FN")}')
print("number of images:", len(image_files))
