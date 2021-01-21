from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from cnn_experiments.yolo.trch_weights import get_weights
from cnn_experiments.yolo.trch_yolonet import YoloNetOrig
from cnn_experiments.window_classifier.model.yolo_datasets import TileImageTestDataset
from cnn_experiments.window_classifier,utils.yolo import yolo_output_to_box

def YoloClass(ABC):
    def __init__(
        self,
        saveweightspath: str,
        channels_in: int
    ) -> None:
        self.saveweightspath = saveweightspath

        ### Yolo parameters
        img_w = 1856
        img_h = 1248
        max_annotations = 14
        anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], [5.319540, 6.116692]]
        nclazz = 1
        box_size = [32, 32]
        weightspath = "/data/old_home_dir/ChrissyF/GFRC/yolov2.weights"
        lambda_c = 5.0
        lambda_no = 0.5
        lambda_cl = 1
        lambda_cf = 1
        n_box = 5

        # Calculate derived parameters
        out_len = 5 + nclazz
        fin_size = n_box * out_len
        grid_w = int(img_w / box_size[1])
        grid_h = int(img_h / box_size[0])
        input_vec = [grid_w, grid_h, n_box, out_len]
        anchors = np.array(anchors)

        # Set up model
        layerlist = get_weights(weightspath)
        self.net = YoloNetOrig(layerlist, fin_size, channels_in)
        
    def inference_on_image(tilez):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        net.load_state_dict(torch.load(self.saveweightspath))
        net.eval()
        tile_dataset = TileImageTestDataset(tilez)
        tileloader = DataLoader(tile_dataset, batch_size=1, shuffle=False)
        boxes_whole_im = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'conf', 'class', 'tile'])
        for idx, tile in enumerate(tileloader):
            tile = tile.to(device)
            y_pred = net(tile)
            y_pred_np = y_pred.data.cpu().numpy()
            boxes_tile = yolo_output_to_box_test(y_pred_np, conf_threshold)
            boxes_tile['tile'] = idx
            boxes_whole_im = pd.concat((boxes_whole_im, boxes_tile), axis=0)

        return boxes_whole_im