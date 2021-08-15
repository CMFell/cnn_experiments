import numpy as np
import torch


class YoloLoss(torch.nn.Module):

    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, outputs, samp_bndbxs, y_true, anchors, scalez, cell_grid, ep, no_obj_thresh):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def reshape_ypred(outputz):
            # reshape outputs to separate anchor boxes
            outputz = outputz.unsqueeze(4)
            outputz = torch.chunk(outputz, 5, dim=3)
            outputz = torch.cat(outputz, dim=4)
            outputz = outputz.transpose(3, 4)
            return outputz

        def split_preds(outputz):
            # split to get individual outputs
            xypred = torch.sigmoid(outputz[..., 0:2])
            whpred = outputz[..., 2:4]
            cfpred = torch.sigmoid(outputz[..., 4])
            clpred = torch.sigmoid(outputz[..., 5:])
            clpred = clpred.squeeze()
            return xypred, whpred, cfpred, clpred

        def split_preds_multi(outputz):
            # split to get individual outputs
            xypred = torch.sigmoid(outputz[..., 0:2])
            whpred = outputz[..., 2:4]
            cfpred = torch.sigmoid(outputz[..., 4])
            # clpred = torch.sigmoid(outputz[..., 5:])
            clpred = torch.nn.functional.softmax(outputz[..., 5:], dim=4)
            # clpred = clpred.squeeze()
            return xypred, whpred, cfpred, clpred

        def create_bndbx_masks(sampbndbxs):
            # get mask of which areas are zero
            wh_gt = sampbndbxs[:, :, 2]
            wh_gt[wh_gt == float('inf')] = 0
            bndbxsmask = torch.gt(wh_gt, 0.0001)
            bndbxsmask4 = bndbxsmask.unsqueeze(2)
            bndbxsmask2 = bndbxsmask.unsqueeze(1)
            bndbxsmask2 = bndbxsmask2.unsqueeze(1)
            bndbxsmask2 = bndbxsmask2.unsqueeze(1)

            return bndbxsmask, bndbxsmask2, bndbxsmask4

        # Convert truth for noones.
        def get_true_wi(sampbndbxs, bndbxsmask4):
            truexywi = sampbndbxs[..., 1:3]
            truewhwi = sampbndbxs[..., 3:5]
            zerosreplace = torch.zeros(truexywi.size())
            zerosreplace = zerosreplace.to(device)
            truexywi = torch.where(bndbxsmask4, truexywi, zerosreplace)
            truewhwi = torch.where(bndbxsmask4, truewhwi, zerosreplace)
            truexywi = truexywi.unsqueeze(1)
            truexywi = truexywi.unsqueeze(1)
            truexywi = truexywi.unsqueeze(1)
            truewhwi = truewhwi.unsqueeze(1)
            truewhwi = truewhwi.unsqueeze(1)
            truewhwi = truewhwi.unsqueeze(1)
            return truexywi, truewhwi

        def get_noones_mask(predxywi, predwhwi, truexywi, truewhwi, bndbxsmask, noobjthresh):

            truewhhalf2 = torch.div(truewhwi, 2.0)
            truemins2 = truexywi - truewhhalf2
            truemaxes2 = torch.add(truexywi, truewhhalf2)
            bndbxsmask2 = bndbxsmask.unsqueeze(5)
            zerosreplace = torch.zeros(truemins2.size())
            zerosreplace = zerosreplace.to(device)
            truemins2 = torch.where(bndbxsmask2, truemins2, zerosreplace)
            truemaxes2 = torch.where(bndbxsmask2, truemaxes2, zerosreplace)

            predxywi = predxywi.unsqueeze(4)
            predwhwi = predwhwi.unsqueeze(4)
            predwhhalf2 = torch.div(predwhwi, 2.)
            predmins2 = predxywi - predwhhalf2
            predmaxes2 = torch.add(predxywi, predwhhalf2)

            intersectmins2 = torch.max(predmins2, truemins2)
            intersectmaxes2 = torch.min(predmaxes2, truemaxes2)
            intersectwh2 = intersectmaxes2 - intersectmins2
            zerosreplace2 = torch.zeros(intersectwh2.size())
            zerosreplace2 = zerosreplace2.to(device)
            intersectwh2 = torch.max(intersectwh2, zerosreplace2)
            intersectareas2 = torch.mul(intersectwh2[..., 0], intersectwh2[..., 1])

            trueareas2 = torch.mul(truewhwi[..., 0], truewhwi[..., 1])
            predareas2 = torch.mul(predwhwi[..., 0], predwhwi[..., 1])

            unionareas2 = torch.add((torch.add(predareas2, trueareas2) - intersectareas2), 0.00001)
            iouscoresall = torch.div(intersectareas2, unionareas2)

            zerosreplace3 = torch.zeros(iouscoresall.size())
            zerosreplace3 = zerosreplace3.to(device)
            iouscoresall = torch.where(bndbxsmask, iouscoresall, zerosreplace3)
            bestious = torch.max(iouscoresall, dim=4)
            bestious = bestious.values
            # print("best iou", round(torch.max(bestious).item(), 2))

            # create masks ones and no ones
            noones = torch.lt(bestious, noobjthresh)

            return noones

        def warm_select(epin, warmmat, truemat):
            ep_chk = torch.zeros(truemat.size())
            ep_chk = ep_chk.fill_(epin)
            ep_chk = torch.lt(ep_chk, 200.0)
            ep_chk = ep_chk.to(device)
            truemat = torch.where(ep_chk, warmmat, truemat)
            return truemat

        def process_ytrue_mat(ytrue, cellgrid, gridtrch, anchorz, epin):
            # get x and y relative to box
            truexy = ytrue[..., 0:2]
            # adjust to relative to whole image
            truexywi = torch.div(torch.add(truexy, cellgrid), gridtrch)
            warmxy = torch.empty(truexy.size()).to(device)
            warmxy = warmxy.fill_(0.5)
            warmxywi = torch.div(torch.add(warmxy, cellgrid), gridtrch)
            #truexy = warm_select(epin, warmxy, truexy)
            #truexywi = warm_select(epin, warmxywi, truexywi)

            # get w and h
            truewhwi = ytrue[..., 2:4]
            # adjust w and h
            truewh = torch.div(torch.mul(truewhwi, gridtrch), anchorz)
            truewh_mask = torch.gt(truewh, 0.000001).type(torch.FloatTensor).to(device)
            truewh = torch.log(torch.add(truewh, 0.000001))
            truewh = torch.mul(truewh, truewh_mask)
            warmwh = torch.zeros(truewh.size()).to(device)
            warmwhwi = torch.ones(truewhwi.size()).to(device)
            warmwhwi = torch.div(torch.mul(warmwhwi, anchorz), gridtrch)
            #truewh = warm_select(epin, warmwh, truewh)
            #truewhwi = warm_select(epin, warmwhwi, truewhwi)

            return truexy, truewh, truexywi, truewhwi

        def get_iou_mat(truexywimat, truewhwimat, predxywi, predwhwi):
            # adjust confidence
            truewhhalf = torch.div(truewhwimat, 2.)
            zeros_replace = torch.zeros(truexywimat.size()).to(device)
            truemins = truexywimat - truewhhalf
            # truemins = torch.max(truemins, zeros_replace)
            ones_replace = torch.ones(truexywimat.size()).to(device)
            truemaxes = torch.add(truexywimat, truewhhalf)
            # truemaxes = torch.min(truemaxes, ones_replace)
            trueareas = truemaxes - truemins
            trueareas = torch.mul(trueareas[..., 0], trueareas[..., 1])

            predwhhalf = torch.div(predwhwi, 2.)
            predmins = predxywi - predwhhalf
            # predmins = torch.max(predmins, zeros_replace)
            predmaxes = torch.add(predxywi, predwhhalf)
            predmaxes = torch.min(predmaxes, ones_replace)

            intersectmins = torch.max(predmins, truemins)
            intersectmaxes = torch.min(predmaxes, truemaxes)
            zeros_replace2 = torch.zeros(intersectmaxes.size())
            zeros_replace2 = zeros_replace2.to(device)
            intersectwh = torch.max((intersectmaxes - intersectmins), zeros_replace2)
            intersectareas = torch.mul(intersectwh[..., 0], intersectwh[..., 1])

            predareas = predmaxes - predmins
            predareas = torch.mul(predareas[..., 0], predareas[..., 1])

            # add a small amount to avoid divide by zero, will later be multiplied by zero
            unionareas = (torch.add(predareas, trueareas) - intersectareas)
            iouscores = torch.div(intersectareas, unionareas)

            return iouscores

        obj_scale = scalez[0]
        no_obj_scale = scalez[1]
        class_scale = scalez[2]
        coord_scale = scalez[3]

        # Reshape predictions
        y_pred = reshape_ypred(outputs)

        # Define basic values
        batchsz, gridh, gridw, ankz, finsiz = y_pred.size()
        grid_trch = torch.from_numpy(np.array([gridw, gridh])).type(torch.FloatTensor)
        grid_trch = grid_trch.to(device)
        anchors1 = anchors.unsqueeze(0)
        anchors1 = anchors1.unsqueeze(0)
        anchors1 = anchors1.unsqueeze(0)
        bndbxs_mask, bndbxs_mask2, bndbxs_mask4 = create_bndbx_masks(samp_bndbxs)

        # Process Predictions
        xy_pred, wh_pred, cf_pred, cl_pred = split_preds_multi(y_pred)
        # mask = torch.ge(cf_pred, 0.5).type(torch.FloatTensor)
        # mask = mask.unsqueeze(4).to(device)
        # print("xy", round(torch.max(xy_pred).item(), 2), round(torch.min(xy_pred).item(), 2),
        #       "wh", round(torch.max(wh_pred).item(), 2), round(torch.min(wh_pred).item(), 2),
        #       "cf", round(torch.max(cf_pred).item(), 2), round(torch.min(cf_pred).item(), 2),
        #       "cl", round(torch.max(cl_pred).item(), 2), round(torch.min(cl_pred).item(), 2))

        # Get predictions on whole image
        pred_xy_wi = torch.div(torch.add(xy_pred, cell_grid), grid_trch)
        pred_wh_wi = torch.div(torch.mul(torch.exp(wh_pred), anchors1), grid_trch)

        # get whole image truths and no ones matrix from list of bound boxes
        true_xy_wi_list, true_wh_wi_list = get_true_wi(samp_bndbxs, bndbxs_mask4)
        no_ones = get_noones_mask(pred_xy_wi, pred_wh_wi, true_xy_wi_list, true_wh_wi_list, bndbxs_mask2, no_obj_thresh)

        # get true values and whole image values from true matrix
        true_xy, true_wh, true_xy_wi_mat, true_wh_wi_mat = process_ytrue_mat(y_true, cell_grid, grid_trch, anchors1, ep)
        # print("true_xy", round(torch.max(true_xy).item(), 2), round(torch.min(true_xy).item(), 2),
        #       "true_wh", round(torch.max(true_wh).item(), 2), round(torch.min(true_wh).item(), 2),
        #       "true_xy_wi_mat", round(torch.max(true_xy_wi_mat).item(), 2),
        #       "true_wh_wi_mat", round(torch.max(true_wh_wi_mat).item(), 2))

        iou_scores = get_iou_mat(true_xy_wi_mat, true_wh_wi_mat, pred_xy_wi, pred_wh_wi)
        # print("iou score", round(torch.max(iou_scores).item(), 2))

        ones = y_true[..., 4]
        # warm_ones = torch.zeros(ones.size()).to(device)
        warm_ones = torch.mul(ones, 0.01)
        #ones = warm_select(ep, warm_ones, ones)

        # print("xywi", round(torch.max(pred_xy_wi).item(), 2), round(torch.min(pred_xy_wi).item(), 2),
        #       "whwi", round(torch.max(pred_wh_wi).item(), 2), round(torch.min(pred_wh_wi).item(), 2))

        loss_conf = iou_scores - cf_pred
        loss_conf = torch.pow(loss_conf, 2)
        loss_conf = torch.mul(loss_conf, ones)
        loss_conf = torch.mul(loss_conf, obj_scale)
        loss_conf = torch.sum(loss_conf)

        zeros_replace6 = torch.zeros(cf_pred.size())
        zeros_replace6 = zeros_replace6.to(device)
        loss_noconf = zeros_replace6 - cf_pred
        loss_noconf = torch.pow(loss_noconf, 2)
        no_ones = no_ones.type(torch.FloatTensor)
        no_ones = no_ones.to(device)
        # warm_noones = torch.zeros(no_ones.size()).type(torch.FloatTensor)
        # warm_noones = warm_noones.to(device)
        # warm_noones = torch.mul(no_ones, 0.01)
        # no_ones = warm_select(ep, warm_noones, no_ones)
        loss_noconf = torch.mul(loss_noconf, no_ones)
        loss_noconf = torch.mul(loss_noconf, no_obj_scale)
        loss_noconf = torch.sum(loss_noconf)

        ones = ones.unsqueeze(4)

        ones_replace = torch.ones(cl_pred.size())
        ones_replace = ones_replace.to(device)
        class_true = y_true[..., 5:]
        class_true = class_true.to(device)
        loss_class = class_true - cl_pred
        loss_class = torch.pow(loss_class, 2)
        loss_class = torch.mul(loss_class, ones)
        loss_class = torch.mul(loss_class, class_scale)
        loss_class = torch.sum(loss_class)

        #warm_ones = warm_ones.fill_(0.01)
        #ones = warm_select(ep, warm_ones, ones)

        loss_xy = torch.pow((true_xy - xy_pred), 2)
        loss_xy = torch.mul(loss_xy, ones)
        loss_xy = torch.mul(loss_xy, coord_scale)
        loss_xy = torch.sum(loss_xy)

        loss_wh = torch.pow((true_wh - wh_pred), 2)
        loss_wh = torch.mul(loss_wh, ones)
        loss_wh = torch.mul(loss_wh, ones)
        loss_wh = torch.sum(loss_wh)

        # outz = [loss_conf, loss_noconf, loss_class, loss_wh, loss_xy]
        loss = loss_conf + loss_noconf + loss_wh + loss_xy + loss_class
        # print("total loss", round(loss.item(), 2),
        #       "conf", round(loss_conf.item(), 2),
        #       "noconf", round(loss_noconf.item(), 2),
        #       "class", round(loss_class.item(), 2),
        #       "size", round(loss_wh.item(), 2),
        #      "centre", round(loss_xy.item(), 2))

        return loss

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputs = torch.randn(4, 10, 20, 30)
outputs = outputs.to(device)
bndbxs = np.array([0, 0.35, 0.3, 0.2, 0.25])
bndbxs_pad = np.empty((13,5))
bndbxs = np.vstack((bndbxs, bndbxs_pad))
bndbxs = np.expand_dims(bndbxs, 0)
bndbxs = np.vstack((bndbxs, bndbxs, bndbxs, bndbxs))
bndbxs = torch.from_numpy(bndbxs).type(torch.FloatTensor)
bndbxs = bndbxs.to(device)
y_true = torch.zeros(4, 10, 20, 5, 6)
y_true[0, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true[1, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true[2, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true[3, 3, 6, 0, :] = torch.from_numpy(np.array([0.5, 0, 0.2, 0.25, 1.0, 1.0]))
y_true = y_true.to(device)

anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = torch.from_numpy(np.array(anchors_in)).type(torch.FloatTensor)
anchors_in = anchors_in.to(device)
scalez = [1, 0.5, 1, 1]
batchsz, gridh, gridw, longout = outputs.size()
cell_x = np.reshape(np.tile(np.arange(gridw), gridh), (1, gridh, gridw, 1))
cell_y = np.reshape(np.repeat(np.arange(gridh), gridw), (1, gridh, gridw, 1))
# combine to give grid
cell_grid = np.tile(np.stack([cell_x, cell_y], -1), [1, 1, 1, 5, 1])
cell_grid = torch.from_numpy(cell_grid).type(torch.FloatTensor)
cell_grid = cell_grid.to(device)

criterion = YoloLoss()
loss = criterion(outputs, bndbxs, y_true, anchors_in, 0.3, scalez, cell_grid, 0)
print(loss)

# x = torch.zeros(10)
# y = 1/x  # tensor with all infinities
# print(y)
# y[y == float('inf')] = 0
# print(y)

"""