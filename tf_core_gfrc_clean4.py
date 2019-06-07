import tensorflow as tf
import pandas as pd
import numpy as np
from my_loss_tf7 import loss_gfrc_yolo, total_loss_calc
from create_dataset2 import BatchGenerator
# from out_box_class_conf import convert_pred_to_output_np
from importweights3 import get_weights, process_layers, random_layers
from tf_gfrc_model import gfrc_model

tf.reset_default_graph()

weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"
filt_in = [3, 32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024,
           1024, 3072, 1024]
n_filters_read = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 1024,
                  1024, 1024, 55]
n_filters = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 1024,
             1024, 1024, 30]
filtersizes = [3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 3, 3, 1]


# Read in file with paths to images and groundtruths
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train384_5pc/'
train_file = base_dir + "gfrc_train.txt"
test_img_in = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_zoom/Z101_Img00083_217.png"
test_rez_out = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_zoom/Z101_Img00083_217.txt"
# set batch size and take first records as batch
n_batch = 1
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
n_anchors = anchors_in.shape[0]
n_classes = 1
out_len = 5 + n_classes
fin_size = out_len * n_anchors
max_gt = 14
train_img_size = (384, 576)
# size_reduction = (16, 16)
size_reduction = (32, 32)
anchors_in_train = np.divide(np.multiply(anchors_in, size_reduction), (train_img_size[1], train_img_size[0]))
anchors_out_train = np.multiply(anchors_in, size_reduction)
learning_rate1 = 0.000001
learning_rate2 = 0.0001
mmtm = 0.9
n_epochs = 1
ini_ep = 0
n_ep = 0
#warmbt = np.ceil(12800. / n_batch)
warmbt = 1600
ini = True


# define values for calculating loss
lambda_cl = 1.0
lambda_no = 0.05
lambda_ob = 10.0
lambda_cd = 1.0
lambda_sz = 1.0
threshold = 0.3
boxy = np.int(np.ceil(train_img_size[0] / size_reduction[0]))
boxx = np.int(np.ceil(train_img_size[1] / size_reduction[1]))
size_out = [boxx, boxy]
maxoutsize = [boxy, boxx, 1, 5 + n_classes, max_gt]
model_dict = {
    'batch_size': n_batch,
    'boxs_x': boxx,
    'boxs_y': boxy,
    'anchors': anchors_in,
    'lambda_coord': lambda_cd,
    'lambda_noobj': lambda_no,
    'lambda_class': lambda_cl,
    'lambda_object': lambda_ob,
    'lambda_size': lambda_sz,
    'n_classes': n_classes,
    'iou_threshold': threshold,
    'n_anchors': n_anchors,
    'num_out': out_len,
    'warmbat': warmbt
}

# read in file names for images and labels
input_file = pd.read_csv(train_file)
image_paths = input_file.img_name
dir_rep = np.repeat(base_dir, image_paths.shape[0])
file_dir = pd.DataFrame(dir_rep, columns=["basedir"])
file_dir = file_dir.basedir
image_paths = file_dir.str.cat(image_paths, sep=None)
image_paths_list = image_paths.tolist()
gt_paths = input_file.gt_details
gt_paths = file_dir.str.cat(gt_paths, sep=None)
gt_paths_list = gt_paths.tolist()
n_pix = len(image_paths)
bat_per_epoch = np.int(np.ceil(n_pix / n_batch))
n_steps = np.int(np.multiply(bat_per_epoch, n_epochs))

paths = np.stack((image_paths, gt_paths), axis=-1)
print(paths.shape)
np.random.shuffle(paths)
print(paths)

batch_dict = {
    'BATCH_SIZE': n_batch,
    'IMAGE_H': train_img_size[0],
    'IMAGE_W': train_img_size[1],
    'TRUE_BOX_BUFFER': max_gt,
    'BOX': n_anchors,
    'N_CLASSES': n_classes,
    'GRID_H': boxy,
    'GRID_W': boxx,
    'ANCHORS': anchors_in
}

tf.logging.set_verbosity(tf.logging.INFO)
img_out = tf.placeholder(tf.float32, shape=(None, None, None, 3))
gt1_out = tf.placeholder(tf.float32, shape=(None, None, None, n_anchors, out_len))
gt2_out = tf.placeholder(tf.float32, shape=(None, 1, 1, 1, max_gt, 4))
ws_in = tf.placeholder(tf.int32)

lay_out = get_weights(weightspath, n_filters_read, filt_in, filtersizes)
biases_dict, weights_dict = process_layers(lay_out, filt_in, n_filters_read, n_filters, filtersizes)
#biases_dict, weights_dict = random_layers(filt_in, n_filters, filtersizes)

y_pred = gfrc_model(img_out, weights_dict, biases_dict)

individual_losses = loss_gfrc_yolo(gt1=gt1_out, gt2=gt2_out, y_pred=y_pred, bat_no=ws_in, dict_in=model_dict,
                                   biasdict=biases_dict, wtdict=weights_dict)
loss = total_loss_calc(individual_losses, dict_in=model_dict)


met_cf_ngt = individual_losses["conf_loss_nogt"]
met_cf_gt = individual_losses["conf_loss_gt"]
met_cnt = individual_losses["cent_loss"]
met_sz = individual_losses["size_loss"]
met_cl = individual_losses["class_loss"]
met_tp = individual_losses["TP"]
met_fp = individual_losses["FP"]
met_fn = individual_losses["FN"]
met_re = individual_losses["Re"]
met_pr = individual_losses["Pr"]
met_fpr = individual_losses["FPR"]
met_t1 = individual_losses["FPR"]
met_t2 = individual_losses["TP"]


tf.summary.scalar('loss', loss)
tf.summary.scalar('met_cf_ngt', met_cf_ngt)
tf.summary.scalar('met_cf_gt', met_cf_gt)
tf.summary.scalar('met_cnt', met_cnt)
tf.summary.scalar('met_sz', met_sz)
tf.summary.scalar('met_cl', met_cl)
tf.summary.scalar('met_tp', met_tp)
tf.summary.scalar('met_fp', met_fp)
tf.summary.scalar('met_fn', met_fn)
tf.summary.scalar('met_re', met_re)
tf.summary.scalar('met_pr', met_pr)
tf.summary.scalar('met_fpr', met_fpr)
tf.summary.scalar('met_t1', met_t1)
tf.summary.scalar('met_t2', met_t2)


def calc_step(opt, btno):
    btno = tf.cast(btno, dtype=tf.float32)
    lr = tf.Variable(opt._lr, name="adam_lr")
    bt1 = tf.Variable(opt._beta1, name="adam_bt1")
    bt2 = tf.Variable(opt._beta2, name="adam_bt2")

    step = lr * tf.sqrt(1. - tf.pow(bt2, btno)) / (1. - tf.pow(bt1, btno))

    return step

#m = tf.Variable()

#lst_vars = []
#for v in tf.global_variables():
#    lst_vars.append(v)

#print(lst_vars)

optimizer = tf.train.AdamOptimizer(learning_rate1, name="Adamopt")


#gvs = optimizer.compute_gradients(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=mmtm, use_nesterov=True)

#learnr = tf.Variable(0.0)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -5., 5.)



#gvs = optimizer.compute_gradients(loss)
#clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
#train = optimizer.apply_gradients(clipped_gradients)


train = optimizer.minimize(loss)

learnr = calc_step(optimizer, ws_in)
tf.summary.scalar('learnr', learnr)

merged = tf.summary.merge_all()

if ini:
    init = tf.global_variables_initializer()

#for v in tf.global_variables():
#    print(v.name, v.dtype, v.shape)

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=400)
#saver = tf.train.Saver(var_list=lst_vars)

with tf.Session() as sess:
    rest_path = "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model_" + str(ini_ep) + ".ckpt"
    if ini:
        sess.run(init)
    else:
        saver.restore(sess, rest_path)
    summary_writer = tf.summary.FileWriter("E:/CF_Calcs/BenchmarkSets/GFRC/tb_log/", tf.get_default_graph())
    train_generator = BatchGenerator(paths, batch_dict)
    tot_bat = 0
    for ep in range(n_epochs):
        cf_ngt = 0
        cf_gt = 0
        cnt_ls = 0
        sz_ls = 0
        cl_ls = 0
        tp_ls = 0
        fp_ls = 0
        fn_ls = 0
        re_ls = 0
        pr_ls = 0
        tl = 0
        tst1 = 1
        ind_in = 0
        for bt in range(bat_per_epoch):
            batch = train_generator.next_batch(ind_in)
            xx = batch[0]
            img_bat = xx['images']
            yy = batch[1]
            gt1_bat = yy['gt1']
            gt2_bat = yy['gt2']
            ind_in = yy['ind']

            input2sess = (train, loss, merged, met_cf_ngt, met_cf_gt, met_cnt, met_sz, met_cl,
                          met_tp, met_fp, met_fn, met_re, met_pr, met_t1, learnr)
            dict4feed = {img_out: img_bat, gt1_out: gt1_bat, gt2_out: gt2_bat, ws_in: tot_bat}
            _, loss_value, summary, outcfngt, outcfgt, outcnt, outsz, outcl, outtp, outfp, outfn, outre, outpr, test1, \
                learnrat = sess.run(input2sess, feed_dict=dict4feed)

            output = sess.run(y_pred, feed_dict=dict4feed)

            #if np.isnan(loss_value):
            np.set_printoptions(threshold=np.inf)
            print(output.shape)


            #print(sess.run(current_lr))
            summary_writer.add_summary(summary, ep * bat_per_epoch + bt)
            cf_ngt += outcfngt
            cf_gt += outcfgt
            cnt_ls += outcnt
            sz_ls += outsz
            cl_ls += outcl
            tp_ls += outtp
            fp_ls += outfp
            fn_ls += outfn
            re_ls += outre
            pr_ls += outpr
            tl += loss_value
            tot_bat += 1

            print("Batch ", bt + 1, " - loss: ", "{0:.2f}".format(loss_value), " - no_gt: ", "{0:.2f}".format(outcfngt),
                  " - gt: ", "{0:.2f}".format(outcfgt), " - cent: ", "{0:.2f}".format(outcnt),
                  " - size: ", "{0:.2f}".format(outsz), " - class: ", "{0:.2f}".format(outcl),
                  " - TP: ", "{0:.2f}".format(outtp), " - FP: ", "{0:.2f}".format(outfp),
                  " - FN: ", "{0:.2f}".format(outfn), " - Test: ", "{0:.2f}".format(test1),
                  " - re: ", "{0:.2f}".format(outre), " - pr: ", "{0:.2f}".format(outpr), "lrat: ", "{0:.8f}".format(learnrat))

        cf_ngt = cf_ngt / bat_per_epoch
        cf_gt = cf_gt / bat_per_epoch
        cnt_ls = cnt_ls / bat_per_epoch
        sz_ls = sz_ls / bat_per_epoch
        cl_ls = cl_ls / bat_per_epoch
        tp_ls = tp_ls
        fp_ls = fp_ls
        fn_ls = fn_ls
        re_ls = re_ls / bat_per_epoch
        pr_ls = pr_ls / bat_per_epoch
        tl = tl / bat_per_epoch
        print("Epoch ", ep + 1, " - loss: ", "{0:.2f}".format(tl), " - no_gt: ", "{0:.2f}".format(cf_ngt),
              " - gt: ", "{0:.2f}".format(cf_gt), " - cent: ", "{0:.2f}".format(cnt_ls),
              " - size: ", "{0:.2f}".format(sz_ls), " - class: ", "{0:.2f}".format(cl_ls),
              " - TP: ", "{0:.2f}".format(tp_ls), " - FP: ", "{0:.2f}".format(fp_ls),
              " - FN: ", "{0:.2f}".format(fn_ls), " - Recall: ", "{0:.2f}".format(re_ls),
              " - Precision: ", "{0:.2f}".format(pr_ls), " - Test: ", "{0:.2f}".format(tst1))

        if ep % 1 == 0 and ep >= 0:
            print("Saving...")
            path2save = "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model_" + str(n_ep) + ".ckpt"
            save_path = saver.save(sess, path2save)
            n_ep += 1
