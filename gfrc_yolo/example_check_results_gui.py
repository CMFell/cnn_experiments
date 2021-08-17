import cv2
import os
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
import tkinter as tk

class Window(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.results_path = 'C:/Users/kryzi/Documents/gfrc_rgb_baseline2_results.csv'
        gfrc_results = pd.read_csv(self.results_path)
        self.gfrc_results_sort = gfrc_results.sort_values(by='filename')
        if 'checked' not in self.gfrc_results_sort.columns:
            self.gfrc_results_sort['checked'] = np.zeros(self.gfrc_results_sort.shape[0])
        if 'manual_result' not in self.gfrc_results_sort.columns:
            self.gfrc_results_sort['manual_result'] = np.repeat("", self.gfrc_results_sort.shape[0])
        basedir = 'C:/Users/kryzi/OneDrive - University of St Andrews/PhD/Data_to_Save/'
        self.pos_folder = basedir + 'test_images/pos/'
        self.neg_folder = basedir + 'test_images/neg/'
        self.init_window()


    # Creation of init_window
    def init_window(self):
        self.imagefile = ""
        self.rowimage = None
        self.idx = int(np.sum(self.gfrc_results_sort['checked']))

        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 50)
        rowxmx = min(7360, row.xmx + 50)
        rowymn = max(0, row.ymn - 50)
        rowymx = min(4912, row.ymx + 50)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        # creating a button instance
        AnimalButton = tk.Button(self, text="Animal", command=self.onClickAnimal)
        NotAnimalButton = tk.Button(self, text="Not Animal", command=self.onClickNot)
        SaveButton = tk.Button(self, text="Save", command=self.onClickSave)
        BackButton = tk.Button(self, text="Back", command=self.onClickBack)

        # placing the button on my window
        AnimalButton.place(x=0, y=250)
        NotAnimalButton.place(x=75, y=250)
        SaveButton.place(x=150, y=250)
        BackButton.place(x=225, y=250)

    def onClickAnimal(self):
        # update details
        self.gfrc_results_sort['checked'].iloc[self.idx] = 1
        self.gfrc_results_sort['manual_result'].iloc[self.idx] = "Animal"
        self.idx = self.idx + 1

        self.imgwindow.config(image="")
        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 50)
        rowxmx = min(7360, row.xmx + 50)
        rowymn = max(0, row.ymn - 50)
        rowymx = min(4912, row.ymx + 50)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

    def onClickNot(self):
        # update details
        self.gfrc_results_sort['checked'].iloc[self.idx] = 1
        self.gfrc_results_sort['manual_result'].iloc[self.idx] = "Not_Animal"
        self.idx = self.idx + 1

        self.imgwindow.config(image="")
        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 50)
        rowxmx = min(7360, row.xmx + 50)
        rowymn = max(0, row.ymn - 50)
        rowymx = min(4912, row.ymx + 50)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

    def onClickSave(self):
        self.gfrc_results_sort.to_csv(self.results_path, index=False)

    def onClickBack(self):
        # update details
        self.idx = self.idx - 1
        self.gfrc_results_sort['checked'].iloc[self.idx] = 0
        self.gfrc_results_sort['manual_result'].iloc[self.idx] = ""

        self.imgwindow.config(image="")
        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 50)
        rowxmx = min(7360, row.xmx + 50)
        rowymn = max(0, row.ymn - 50)
        rowymx = min(4912, row.ymx + 50)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

root = tk.Tk()

# size of the window
root.geometry("400x300")

app = Window(root)
root.mainloop()
