"code adapted from https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e"
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.cfno = []
        self.cf = []
        self.ct = []
        self.sz = []
        self.cl = []
        self.fp = []
        self.tp = []
        self.fn = []

        self.fig = plt.figure()

        self.logs = []

        plt.ion()
        #fig, ax1 = plt.subplots()
        #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #fig.legend(["loss", "TP", "FP", "FN"])
        #ax1.set_ylim([0, 100])
        #ax1.set_xlabel('epoch')
        #ax1.set_ylabel('loss')
        #ax1.plot(self.x, self.losses, color='r')
        #ax2.set_ylabel('conditions')
        #fig.tight_layout()

        #ax2.plot(self.x, self.tp, color='m')
        #plt.ylim([0,150])
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.cfno.append(logs.get('metric_yolo_cfno'))
        self.cf.append(logs.get('metric_yolo_cf'))
        self.ct.append(logs.get('metric_yolo_ct'))
        self.sz.append(logs.get('metric_sz'))
        self.cl.append(logs.get('metric_yolo_cl'))
        self.tp.append(logs.get('metric_TP'))
        self.fp.append(logs.get('metric_FP'))
        self.fn.append(logs.get('metric_FN'))
        self.i += 1
        clear_output(wait=False)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.sz, label="size")
        #ax1.plot(self.x, self.losses, color='r')


        #ax2.plot(self.x, self.tp, color='g')
        #plt.ax2.plot(self.x, self.fp, color='c')

        #plt.plot(self.x, self.cfno, label="no object")
        #plt.plot(self.x, self.cf, label="object")
        #plt.plot(self.x, self.ct, label="centre")
        #plt.plot(self.x, self.sz, label="size")
        #plt.plot(self.x, self.cl, label="class")
        plt.pause(0.001)

plot_losses = PlotLosses()