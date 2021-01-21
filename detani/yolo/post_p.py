import numpy as np

def calc_iou_pp(gt, dets, thresh):
    area_det = np.multiply(np.subtract(dets[:, 2], dets[:, 0]), np.subtract(dets[:, 3], dets[:, 1]))
    area_true = (gt[2] - gt[0]) * (gt[3] - gt[1])
    inter_xmaxs = np.minimum(dets[:, 2], gt[2])
    inter_ymaxs = np.minimum(dets[:, 3], gt[3])
    inter_xmins = np.maximum(dets[:, 0], gt[0])
    inter_ymins = np.maximum(dets[:, 1], gt[1])
    inter_xsize = np.maximum(np.subtract(inter_xmaxs, inter_xmins), 0)
    inter_ysize = np.maximum(np.subtract(inter_ymaxs, inter_ymins), 0)
    inter_area = np.multiply(inter_xsize, inter_ysize)
    union_area = np.subtract(np.add(area_true, area_det), inter_area)
    iou = np.divide(inter_area, union_area)
    iou_max = np.max(iou)
    if iou_max >= thresh:
        TP = 1
        mask = np.logical_not(np.equal(iou, iou_max))
        dets_out = dets[mask, :]
        FN = 0
    else:
        TP = 0
        dets_out = dets
        FN = 1
    return TP, FN, dets_out

def post_p(gt_box, det_box, thresh):
    TPz = 0
    FNz = 0
    if det_box.shape[0] > 1:
        if len(gt_box.shape) > 1:
            for gg in range(gt_box.shape[0]):
                gt_row = gt_box[gg]
                tp, fn, det_box = calc_iou_pp(gt_row, det_box, thresh)
                TPz += tp
                FNz += fn
        else:
            tp, fn, det_box = calc_iou_pp(gt_box, det_box, thresh)
            TPz += tp
            FNz += fn
        FPz = det_box.shape[0]
    else:
        TPz += 0
        FNz += gt_box.shape[0]
        FPz = 0
    return TPz, FNz, FPz