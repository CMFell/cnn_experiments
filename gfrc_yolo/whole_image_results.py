import cv2
import numpy as np
from pathlib import Path
import pandas as pd

from postprocess.calculate_metrics import conf_mat_raw, 
from postprocess.image_process import get_image_file_names, predict_on_images
from postprocess.results_process import remove_on_size, calculate_threshold_results, single_threshold_results, get_image_results
from postprocess.truth_process import process_truths
from postprocess.visualise_results import draw_res, get_fnz, get_fpz, create_mosaic


# Select base directory where downloaded image files and weights saved 
basedir = 'E:/GFRC_data/'
# type of input valid or train
settype = 'valid'
# Select from one of two pretrained models or other for a newly trained one
# If using an other model you will need to set further parameters in lines 64 to 79
model_to_use = ['without_metadata', 'with_metadata', 'other'][0]

# Use the following flags to set what results to calculate
# Each step need to be carried out in order
# Each step saves results and by setting to False earlier saved results can be read in instead of recalculating

# 1. Use the model to detect animals in all images
predict_results = True
# 2. Calculate the metrics at different probability thresholds so you can pick the best
results_all_thresholds = True
# 3. Calculate the combined results for all images for the chosen threshold
calculate_results_for_threshold = True
# 4. Calculate results for each image separately for the chosen threshold
calculate_image_results = True
# 5. Draw example results, draw results on images in image list, create fp and fn mosaics
draw_example_results = True
# 6. Calculate and save confusion matrices
calcuate_confusion_matrices = True

# Set list of example images to draw results on
if settype == 'valid':  
    image_list_to_draw = ['Z162_Img09635.jpg', 'Z138_Img02888.jpg', 'Z247_Img14318.jpg', 'Z124_Img13083.jpg']
else:
    image_list_to_draw = ['Z138_Img02888.jpg', 'Z120_Img11234.jpg', 'Z124_Img13083.jpg']

### select model to use
if model_to_use == 'without_metadata':
    binmulti = 'bin'
    name_out = 'rgb_bin'
    save_dir = basedir + 'output/' name_out + "/"
    saveweightspath = basedir + "weights/rgb_binary_weights.pt"
    channels_in = 3
    gfrc_conf_threshold = 0.2
    meta_end = False
    col_list = None
elif model_to_use == 'with_metadata':
    binmulti = 'bin'
    name_out = 'rgb_bin'
    save_dir = basedir + 'output/' name_out + "/"
    saveweightspath = basedir + "weights/combined_ci_weights.pt"
    channels_in = 3
    gfrc_conf_threshold = 0.25
    meta_end = True
    col_list = ['lowerci', 'upperci']
    channels_in = channels_in + len(col_list)
else:
    binmulti = 'bin'
    name_out = 'rgb_bin_retrained'
    save_dir = basedir + 'output/' name_out + "/"
    # epoch number of weights to use
    nn = 149
    saveweightspath = save_dir + "testing_save_" + str(nn) + ".pt"
    # RGB so 3 channels change if not
    channels_in = 3
    # set chosen confidence threshold
    gfrc_conf_threshold = 0.5
    # add metadata at start or end
    meta_end = False
    # set list of columns to use for metadata
    col_list = None
    channels_in = channels_in + len(col_list)


if binmulti == 'bin':
    nclazz = 1
    title = "GFRC binary confusion matrix"
    labels = [0, 1]
    categories = ["none", "animal"]
else:
    nclazz = 6
    title = "GFRC multiclass confusion matrix"
    labels = [0, 1, 2, 3, 4, 5, 6]
    categories = ["none", "bok", "oryx", "kudu", "zeb", "ost", "unid"]
    # not animal 0 code=none
    # springbok 1 (0+1) code=bok
    # oryx 2 (1+1) code=oryx
    # kudu 3 (2+1) code=kudu
    # zebra 4 (3+1) code=zeb
    # ostrich 5 (4+1) code=ost
    # unidentified 6 (5+1) code=unid

whole_image_dir = basedir + settype + "_images/"
truth_file = whole_image_dir + settype + "_GFRC_bboxes_" + binmulti + ".csv"
output_save_dir = save_dir + settype + "_results/"
Path(output_save_dir).mkdir(parents=True, exist_ok=True)

# Get true positions for comparison
gfrc_truth = pd.read_csv(truth_file)
gfrc_truth = process_truths(gfrc_truth, 'gfrc')

# process image files
if predict_results:
    image_files = get_image_file_names(whole_image_dir)
    yolo_model = YoloClass(wtpath=saveweightspath, channels=channels_in, nclazz=nclazz, meta_cols=col_list, meta_end=meta_end)
    gfrc_windows_in = predict_on_images(image_files, whole_image_dir, yolo_model, basedir)
    windows_filename = name_out + '_windows.csv'
    gfrc_windows_in.to_csv(output_save_dir + windows_filename, index=False)
else:
    gfrc_windows = pd.read_csv(output_dir + windows_filename)

if results_all_thresholds:
    # remove too big and too small detections
    gfrc_windows = remove_on_size(gfrc_windows_in)
    gfrc_metrics = calculate_threshold_results(gfrc_windows, gfrc_truth, whole_image_dir, 0.25, 0.25, 'mean', 0.05, 96)
    gfrc_metrics.to_csv(output_save_dir + name_out + "_metrics.csv", index=False)
else:
    gfrc_metrics = pd.read_csv(output_save_dir + name_out + "_metrics.csv")

if calculate_results_for_threshold:
    gfrc_results = single_threshold_results(gfrc_windows, gfrc_truth, whole_image_dir, gfrc_conf_threshold, 0.25, 0.25, 'mean')
    gfrc_results.to_csv(output_save_dir + name_out + "_results.csv", index=False)
else:
    gfrc_results = pd.read_csv(output_save_dir + name_out + "_results.csv")

recall = np.sum(gfrc_results.confmat == "TP") / (np.sum(gfrc_results.confmat == "TP") + np.sum(gfrc_results.confmat == "FN"))
FPPI = np.sum(gfrc_results.confmat == "FP") / 316
print("recall: ", recall, " , FPPI: ", FPPI)

if calculate_image_results:
    gfrc_image_results = get_image_stats(gfrc_results)
    gfrc_image_results.to_csv(output_save_dir + name_out + "_image_results.csv", index=False)
else:
    gfrc_image_results = pd.read_csv(output_save_dir + name_out + "_image_results.csv")

if draw_example_results:
    draw_res(gfrc_results, image_files_to_draw, whole_image_dir, output_save_dir, name_out)
    gfrc_fnz = get_fnz(gfrc_results, whole_image_dir, output_save_dir, 'gfrc', experiment)
    gfrc_fpz = get_fpz(gfrc_results, whole_image_dir, output_save_dir, 'gfrc', experiment)
    fnz_combined = create_mosaic(gfrc_fnz, (4, 8), (60*2, 60*2))
    fnz_combined = cv2.cvtColor(fnz_combined, cv2.COLOR_BGR2RGB)
    fnz_name = 'fnz_mosaic_' + experiment + '.jpg'
    cv2.imwrite(str(output_save_dir + fnz_name), fnz_combined)
    fpz_combined = create_mosaic(gfrc_fpz, (4, 8), (60*2, 60*2))
    fpz_combined = cv2.cvtColor(fpz_combined, cv2.COLOR_BGR2RGB)
    fpz_name = 'fpz_mosaic_' + experiment + '.jpg'
    cv2.imwrite(str(output_save_dir + fpz_name), fpz_combined)

if calcuate_confusion_matrices:
    gfrc_results['tru_class_bin'] = np.array(gfrc_results.tru_class > 0, dtype=np.int)
    cm_gfrc = conf_mat_raw(gfrc_results.tru_class_bin, gfrc_results.pred_class, labels)
    affix = name_out + '-' + binmulti
    save_conf_mat_plot_no_title(cm_gfrc, categories, output_save_dir, affix)




