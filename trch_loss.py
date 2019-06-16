import torch
import numpy as np

class YoloLoss(torch.nn.Module):

    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, outputs, samp_bndbxs, y_true, anchors, no_obj_thresh, scalez, cell_grid):
        # reshape outputs to separate anchor boxes
        outputs = outputs.unsqueeze(4)
        outputs = torch.chunk(outputs, 5, dim=3)
        outputs = torch.cat(outputs, dim=4)
        outputs = outputs.transpose(4, 3)
        # split to get individual outputs
        xy_pred = torch.sigmoid(outputs[..., 0:2])
        wh_pred = outputs[..., 2:4]
        cf_pred = torch.sigmoid(outputs[..., 4])
        cl_pred = torch.sigmoid(outputs[..., 5:])
        cl_pred = cl_pred.squeeze()

        ### get mask of which areas are zero
        wh_gt = samp_bndbxs[:, :, 2]
        wh_gt[wh_gt == float('inf')] = 0
        bndbxs_mask = torch.gt(wh_gt, 0.0001)
        bndbxs_mask4 = bndbxs_mask.unsqueeze(2)
        bndbxs_mask2 = bndbxs_mask.unsqueeze(1)
        bndbxs_mask2 = bndbxs_mask2.unsqueeze(1)
        bndbxs_mask2 = bndbxs_mask2.unsqueeze(1)

        ### get cell grid
        batchsz, gridh, gridw, ankz, finsiz = outputs.size()
        grid_trch = torch.from_numpy(np.array([gridw, gridh])).type(torch.FloatTensor)
        anchors1 = anchors.unsqueeze(0)
        anchors1 = anchors1.unsqueeze(0)
        anchors1 = anchors1.unsqueeze(0)
        pred_xy_wi = torch.div(torch.add(xy_pred, cell_grid), grid_trch)
        pred_wh_wi = torch.div(torch.mul(torch.exp(wh_pred), anchors1), grid_trch)
        ### Convert truth for noones.
        true_xy_wi = samp_bndbxs[..., 1:3]
        true_wh_wi = samp_bndbxs[..., 3:5]
        true_xy_wi = torch.where(bndbxs_mask4, true_xy_wi, torch.zeros(true_xy_wi.size()))
        true_wh_wi = torch.where(bndbxs_mask4, true_wh_wi, torch.zeros(true_xy_wi.size()))
        true_xy_wi = true_xy_wi.unsqueeze(1)
        true_xy_wi = true_xy_wi.unsqueeze(1)
        true_xy_wi = true_xy_wi.unsqueeze(1)
        true_wh_wi = true_wh_wi.unsqueeze(1)
        true_wh_wi = true_wh_wi.unsqueeze(1)
        true_wh_wi = true_wh_wi.unsqueeze(1)

        true_wh_half2 = torch.div(true_wh_wi, 2.0)
        true_mins2 = true_xy_wi - true_wh_half2
        true_maxes2 = torch.add(true_xy_wi, true_wh_half2)

        pred_xy_wi1 = pred_xy_wi.unsqueeze(4)
        pred_wh_wi1 = pred_wh_wi.unsqueeze(4)

        pred_wh_half2 = torch.div(pred_wh_wi1, 2.)
        pred_mins2 = pred_xy_wi1 - pred_wh_half2
        pred_maxes2 = torch.add(pred_xy_wi1, pred_wh_half2)

        bndbxs_mask3 = bndbxs_mask2.unsqueeze(5)
        #bndbxs_mask3 = torch.cat([bndbxs_mask3, bndbxs_mask3], dim=5)
        zeros_replace = torch.zeros(true_mins2.size())
        true_mins3 = torch.where(bndbxs_mask3, true_mins2, zeros_replace)
        true_maxes3 = torch.where(bndbxs_mask3, true_maxes2, zeros_replace)
        #true_mins3 = true_mins3.double()
        #true_maxes3 = true_maxes3.double()
        intersect_mins2 = torch.max(pred_mins2, true_mins3)
        intersect_maxes2 = torch.min(pred_maxes2, true_maxes3)
        intersect_wh2 =  intersect_maxes2 - intersect_mins2
        zeros_replace2 = torch.zeros(intersect_wh2.size())
        intersect_wh2 = torch.max(intersect_wh2, zeros_replace2)
        intersect_areas2 = np.multiply(intersect_wh2[..., 0], intersect_wh2[..., 1])

        true_areas2 = torch.mul(true_wh_wi[..., 0], true_wh_wi[..., 1])
        pred_areas2 = torch.mul(pred_wh_wi1[..., 0], pred_wh_wi1[..., 1])

        union_areas2 = torch.add((torch.add(pred_areas2, true_areas2) - intersect_areas2), 0.00001)
        iou_scores_all = torch.div(intersect_areas2, union_areas2)

        zeros_replace3 = torch.zeros(iou_scores_all.size())
        iou_scores_all = torch.where(bndbxs_mask2, iou_scores_all, zeros_replace3)
        best_ious = torch.max(iou_scores_all, dim=4)
        best_ious = best_ious.values

        # create masks ones and no ones
        noones = torch.lt(best_ious, no_obj_thresh)

        # get x and y relative to whole image
        true_box_xy_wi = y_true[..., 0:2]
        # adjust to relative to box
        true_box_xy = torch.div(torch.add(true_box_xy_wi, cell_grid), grid_trch)

        # get w and h
        true_box_wh_wi = y_true[..., 2:4]
        # adjust w and h
        true_box_wh = torch.div(torch.mul(true_box_wh_wi, grid_trch), anchors1)
        true_box_wh = torch.log(torch.add(true_box_wh, 0.000001))

        # adjust confidence
        true_wh_half = torch.div(true_box_wh_wi, 2.)
        true_mins = true_box_xy_wi - true_wh_half
        true_maxes = torch.add(true_box_xy_wi, true_wh_half)
        true_areas = torch.mul(true_box_wh_wi[..., 0], true_box_wh_wi[..., 1])

        pred_wh_half = torch.div(pred_wh_wi, 2.)
        pred_mins = pred_xy_wi - pred_wh_half
        pred_maxes = torch.add(pred_xy_wi, pred_wh_half)

        intersect_mins = torch.max(pred_mins, true_mins)
        intersect_maxes = torch.min(pred_maxes, true_maxes)
        zeros_replace4 = torch.zeros(intersect_maxes.size())
        intersect_wh = torch.max((intersect_maxes - intersect_mins), zeros_replace4)
        intersect_areas = torch.mul(intersect_wh[..., 0], intersect_wh[..., 1])

        pred_areas = torch.mul(pred_wh_wi[..., 0], pred_wh_wi[..., 1])

        # add a small amount to avoid divide by zero, will later be multiplied by zero
        union_areas = torch.add((torch.add(pred_areas, true_areas) - intersect_areas), 0.00001)
        iou_scores = torch.div(intersect_areas, union_areas)

        ones = y_true[..., 4]

        obj_scale = scalez[0]
        no_obj_scale = scalez[1]
        class_scale = scalez[2]
        coord_scale = scalez[3]

        loss_conf = iou_scores - cf_pred
        loss_conf = torch.pow(loss_conf, 2)
        loss_conf = torch.mul(loss_conf, ones)
        loss_conf = torch.mul(loss_conf, obj_scale)
        loss_conf = torch.sum(loss_conf)

        zeros_replace5 = torch.zeros(cf_pred.size())
        loss_noconf = zeros_replace5 - cf_pred
        loss_noconf = torch.pow(loss_noconf, 2)
        noones = noones.type(torch.FloatTensor)
        loss_noconf = torch.mul(loss_noconf, noones)
        loss_noconf = torch.mul(loss_noconf, no_obj_scale)
        loss_noconf = torch.sum(loss_noconf)

        ones_replace = torch.ones(cl_pred.size())
        loss_class = ones_replace - cl_pred
        loss_class = torch.pow(loss_class, 2)
        loss_class = torch.mul(loss_class, ones)
        loss_class = torch.mul(loss_class, class_scale)
        loss_class = torch.sum(loss_class)

        ones = ones.unsqueeze(4)

        loss_xy = torch.pow((true_box_xy - xy_pred), 2)
        loss_xy = torch.mul(loss_xy, ones)
        loss_xy = torch.mul(loss_xy, coord_scale)
        loss_xy = torch.sum(loss_xy)

        loss_wh = torch.pow((true_box_wh - wh_pred), 2)
        loss_wh = torch.mul(loss_wh, ones)
        loss_wh = torch.mul(loss_wh, ones)
        loss_wh = torch.sum(loss_wh)

        #outz = [loss_conf, loss_noconf, loss_class, loss_wh, loss_xy]
        loss = loss_conf + loss_noconf + loss_class + loss_wh + loss_xy

        return loss

"""
outputs = torch.randn(4, 10, 20, 30)
bndbxs = np.array([0, 0.35, 0.3, 0.2, 0.25])
bndbxs_pad = np.empty((13,5))
bndbxs = np.vstack((bndbxs, bndbxs_pad))
bndbxs = np.expand_dims(bndbxs, 0)
bndbxs = np.vstack((bndbxs, bndbxs, bndbxs, bndbxs))
bndbxs = torch.from_numpy(bndbxs).type(torch.FloatTensor)
y_true = torch.zeros(4, 10, 20, 5, 6)
y_true[0, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true[1, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true[2, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true[3, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))

anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = torch.from_numpy(np.array(anchors_in)).type(torch.FloatTensor)
scalez = [1, 0.5, 1, 1]
batchsz, gridh, gridw, longout = outputs.size()
cell_x = np.reshape(np.tile(np.arange(gridw), gridh), (1, gridh, gridw, 1))
cell_y = np.reshape(np.repeat(np.arange(gridh), gridw), (1, gridh, gridw, 1))
# combine to give grid
cell_grid = np.tile(np.stack([cell_x, cell_y], -1), [1, 1, 1, 5, 1])
cell_grid = torch.from_numpy(cell_grid).type(torch.FloatTensor)

criterion = YoloLoss()
loss = criterion(outputs, bndbxs, y_true, anchors_in, 0.3, scalez, cell_grid)
print(loss)

#x = torch.zeros(10)
#y = 1/x  # tensor with all infinities
#print(y)
#y[y == float('inf')] = 0
#print(y)

"""