__author__ = "Dani Martinez"
__copyright__ = "Copyright 2025, Moblanc Robotics & Cornell University"
__credits__ = ["Dani Martinez"]
__license__ = "Apache 2.0"
__version__ = "0.5"
__maintainer__ = "Dani Martinez"
__email__ = "dani.martinez@moblancrobotics.com"
__status__ = "Production"

import time
from datetime import datetime
import argparse
import os
from pathlib import Path
from threading import Thread, Event
import statistics
import math

import onnxruntime as ort
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import msgpack
import msgpack_numpy as mnp
mnp.patch()



import cpp_functions.leaf_masking as leaf_masking

# Useful colored strings
hwarning = "\033[93m[WARNING]\033[0m: "
herror = "\033[91m[ERROR]\033[0m: "

CPU_BACKEND = "CPUExecutionProvider"
GPU_BACKEND = "DmlExecutionProvider"

N_SAMPLES_X_TRAY = 351

MASKING_RL_TH = 0.2         # Relative threshold used for leaf masking algorithm

SUBIMAGE_HEIGHT = 224
SUBIMAGE_WIDTH = 224

GUI_UPDATE_FREQ = 1000 # ms


class ReturnableThread(Thread):
    # This class is a subclass of Thread that allows the thread to return a value.
    def __init__(self, target, args):
        Thread.__init__(self, args=args)
        self.target = target
        self.args = args
        self.result = None
    
    def run(self) -> None:
        self.result = self.target(*self.args)


class CNNAnalyzerGUI(tk.Frame):
    def __init__(self,
                 master,
                 model_path
    ):
        tk.Frame.__init__(self, master=master)

        # -------------- Internal Variables
        self._win = master
        self._model_path = model_path
        self._exp_dirpath = None
        self._expdata = None    # Includes name, nimages, imagepaths, results

        self._date_ind = 0
        self._tray_ind = 0
        self._img_ind = 0


        # Thread shared data (needs mutable lists)
        self._etc = ["00:00:00"]
        self._progress = [0.0]
        self._interrupt = [False]

        self._tprocess = None


        # -------------- GUI Elements

        self._bt_open = tk.Button(text="Open", command=self._open_callback, font=("Arial 15"))
        self._bt_open.grid(row=0, column=0, rowspan=1, padx=(20, 10), pady=(5, 10), sticky="news")

        self._bt_start = tk.Button(text="Start analysis", command=self._start_callback, font=("Arial 15"))
        self._bt_start.grid(row=0, column=1, rowspan=1, padx=(10, 10), pady=(5, 10), sticky="news")
        self._bt_start["state"] = tk.DISABLED

        self._bt_stop = tk.Button(text="Stop analysis", command=self._stop_callback, font=("Arial 15"))
        self._bt_stop.grid(row=0, column=2, rowspan=1, padx=(10, 20), pady=(5, 10), sticky="news")
        self._bt_stop["state"] = tk.DISABLED

        # ------------ Create text labels ----------------
        bg_frame = tk.Frame(bd=0, borderwidth=2, highlightthickness=1, highlightbackground="black", background="white")
        bg_frame.grid(row=1, column=0, columnspan=3, padx=(20, 20), sticky="news")

        # -- Loaded CNN model --
        cnnmodel = tk.StringVar()
        cnnmodel.set("CNN model:")
        l_cnnmodel = tk.Label(bg_frame,textvariable=cnnmodel,
                              font=("Arial 12 bold"),
                              anchor=tk.CENTER,
                              justify="left",
                              bg="#FFFFFF"
                              )
        l_cnnmodel.grid(row=1, column=0, columnspan=1, sticky="w")

        self._t_expname = tk.StringVar()
        self._l_expname = tk.Label(bg_frame,textvariable=self._t_expname,
                                   font=("Arial 12"),
                                   anchor=tk.CENTER,
                                   justify="center",
                                   width=25,
                                   bg="#FFFFFF"
                                   )
        self._l_expname.grid(row=1, column=1, columnspan=2, sticky="nsew")
        self._t_expname.set(os.path.basename(self._model_path))
        # ---------------------------

        # -- Experiment Name --
        expname = tk.StringVar()
        expname.set("Experiment name:")
        l_expname = tk.Label(bg_frame,textvariable=expname,
                                   font=("Arial 12 bold"),
                                   anchor=tk.CENTER,
                                   justify="left",
                                   bg="#FFFFFF"
                                   )
        l_expname.grid(row=2, column=0, columnspan=1, sticky="w")

        self._t_expname = tk.StringVar()
        self._l_expname = tk.Label(bg_frame,textvariable=self._t_expname,
                                   font=("Arial 12"),
                                   anchor=tk.CENTER,
                                   justify="left",
                                   width=25,
                                   bg="#FFFFFF"
                                   )
        self._l_expname.grid(row=2, column=1, columnspan=2, sticky="nsew")
        self._t_expname.set("- -")
        # ---------------------------

        # -- Nº images --
        nimages = tk.StringVar()
        nimages.set("Nº images:")
        l_nimages = tk.Label(bg_frame,
                             textvariable=nimages,
                             font=("Arial 12 bold"),
                             anchor=tk.CENTER,
                             justify="left",
                             bg="#FFFFFF"
                              )
        l_nimages.grid(row=3, column=0, columnspan=1, sticky="w")

        self._t_nimages = tk.StringVar()
        self._l_nimages = tk.Label(bg_frame,
                                   textvariable=self._t_nimages,
                                   font=("Arial 12"),
                                   anchor=tk.CENTER,
                                   justify="left",
                                   width=25,
                                   bg="#FFFFFF"
                                   )
        self._l_nimages.grid(row=3, column=1, columnspan=2, sticky="nsew")
        self._t_nimages.set("- -")
        # ---------------------------

        # -- Nº Trays --
        ntrays = tk.StringVar()
        ntrays.set("Nº trays:")
        l_ntrays = tk.Label(bg_frame,
                            textvariable=ntrays,
                            font=("Arial 12 bold"),
                            anchor=tk.CENTER,
                            justify="left",
                            bg="#FFFFFF"
                                   )
        l_ntrays.grid(row=4, column=0, columnspan=1, sticky="w")

        self._t_ntrays = tk.StringVar()
        self._l_ntrays = tk.Label(bg_frame,
                                  textvariable=self._t_ntrays,
                                  font=("Arial 12"),
                                  anchor=tk.CENTER,
                                  justify="left",
                                  width=25,
                                  bg="#FFFFFF"
                                  )
        self._l_ntrays.grid(row=4, column=1, columnspan=2, sticky="nsew")
        self._t_ntrays.set("- -")
        # ---------------------------

        # -- Nº timepoints --
        ntimep = tk.StringVar()
        ntimep.set("Nº timepoints:")
        l_ntimep = tk.Label(bg_frame,
                            textvariable=ntimep,
                            font=("Arial 12 bold"),
                            anchor=tk.CENTER,
                            justify="center",
                            bg="#FFFFFF"
                            )
        l_ntimep.grid(row=5, column=0, columnspan=1, sticky="w")

        self._t_ntimep = tk.StringVar()
        self._l_ntimep = tk.Label(bg_frame,
                                  textvariable=self._t_ntimep,
                                  font=("Arial 12"),
                                  anchor=tk.CENTER,
                                  justify="left",
                                  width=25,
                                  bg="#FFFFFF"
                                  )
        self._l_ntimep.grid(row=5, column=1, columnspan=2, sticky="nsew")
        self._t_ntimep.set("- -")
        # ---------------------------


        self._pbar = ttk.Progressbar(orient=tk.HORIZONTAL, maximum=1.0)
        self._pbar.grid(row=6, column=0, columnspan=3, padx=(20, 20), pady=(10,0), sticky="nsew")
        self._pbar['value'] = 0.0

        # Create text widget and specify size.
        self._t_etc = tk.StringVar()
        self._l_etc = tk.Label(textvariable=self._t_etc,
                               font=("Arial 14"),
                               anchor=tk.CENTER,
                               justify="center",
                               )
        self._l_etc.grid(row=7, column=0, columnspan=3, padx=(20, 20), pady=(0, 5),sticky="news")

        percent = 0
        self._t_etc.set(f"{percent}% - ETC 00:00:00")

        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self._close_callback)


    def _open_callback(self):
        expdir = filedialog.askdirectory(title='Open an experiment folder',
                                         mustexist=True)
        if len(expdir):
            # Check if experiment already have results file in it
            if os.path.exists(Path(expdir)/"results.msgpack"):
                answ = messagebox.askyesno(message="This experiment already have results data in it.\nOverwrite?", title="Overwrite?")
                if not answ:
                    self._clear_expdata()
                    return

            # Load experiment
            self._load_experiment(expdir)

    def _close_callback(self):
        self.winfo_toplevel().destroy()
        return

        # TODO: Check if analysis is in progress and ask if so
        answ = messagebox.askyesno(message="Exit?", title="Exiting...")
        print(answ)
        if not answ:
            return
        self.winfo_toplevel().destroy()

    def _load_experiment(self, expdir):

        self._expdata = self._get_expdata(expdir)

        if self._expdata is None:
            self._clear_expdata()
            return
        
        # Check how many unique trays
        unique_trays = {}
        for d in self._expdata["samples"]:
            for t in self._expdata["samples"][d]:
                if t not in unique_trays:
                    unique_trays[t] = True
        
        total_unique_trays = len(unique_trays.keys())
        n_timep = len(self._expdata["samples"])

        self._t_expname.set(self._expdata["name"])
        self._t_nimages.set(str(self._expdata["nimages"]))
        self._t_ntrays.set(str(total_unique_trays))
        self._t_ntimep.set(str(n_timep))
        self._bt_start["state"] = tk.NORMAL


    def _get_expdata(self, expdir):
        n_images = 0
        img_files = {}
        results = {}

        expdir = Path(expdir)

        datefolders = [f for f in os.scandir(expdir) if f.is_dir()]

        if len(datefolders) == 0:
            print(herror+"Experiment folder does not have any timepoint sub-folder!")
            return None

        date_list = []
        for d in datefolders:
            date_str = d.name.split("_")[0]
            date_list.append(datetime.strptime(date_str, '%m-%d-%Y'))

        ind_dates = [i for i, x in sorted(enumerate(date_list), key=lambda x: x[1])]
        datefolders = [datefolders[i] for i in ind_dates]

        for d in datefolders:
            img_files[d.name] = {}
            results[d.name] = {}

            trayfolders = [f for f in os.scandir(d) if f.is_dir()]
            if len(trayfolders) == 0:
                continue

            #TODO: Check trayfolder name is scoremap? if so, skip it?
            
            for t in trayfolders:
                img_files[d.name][t.name] = [None]*N_SAMPLES_X_TRAY
                results[d.name][t.name] = [None]*N_SAMPLES_X_TRAY

                imagefiles = [f for f in os.scandir(t) if f.name.endswith(".png")]
                for i in imagefiles:
                    nsample = int(i.name.split('-')[0])
                    img_files[d.name][t.name][nsample-1] = i.name
                    n_images += 1
        
        exp_data = {}
        exp_data["name"] = expdir.parts[-1]
        exp_data["path"] = expdir
        exp_data["nimages"] = n_images
        exp_data["samples"] = img_files
        exp_data["results"] = results # This might be useless
        return exp_data

    def _clear_expdata(self):
        self._t_expname.set("- -")
        self._t_nimages.set("- -")
        self._t_ntrays.set("- -")
        self._t_ntimep.set("- -")
        self._expdata = None
        self._pbar.step(0)
        self._t_etc.set(f"0% - ETC 00:00:00")

    def _start_callback(self): # Analysis process into a Separated thread

        self._bt_start["state"] = tk.DISABLED
        self._bt_open["state"] = tk.DISABLED
        self._bt_stop["state"] = tk.NORMAL
        
        self._tprocess = ReturnableThread(target=self._process_analysis, args=(self._expdata,
                                                                            self._model_path,
                                                                            self._etc,
                                                                            self._progress,
                                                                            self._interrupt))
        self._tprocess.start()

        self._win.after(GUI_UPDATE_FREQ, self._update_progress)
        pass


    def _stop_callback(self):
        self._interrupt[0] = True

    @staticmethod
    def _process_analysis(expdata, model_path, etc_str, progress, stop) -> dict:

        def read_image(img_path):
            """
                NOTE: This function is implemented to read images from an UTF-8 path string
            """
            stream = open(img_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED) # Returns BGR image

        def isOnFocus(subimg_mask): # TODO: test this
            mask_ratio = cv2.mean(subimg_mask)[0] / 255
            return True if mask_ratio > 0.7 else False

        def compute_sample(sess, img_path, out_pos):
            # Get input and output tensor names to use later when inferencing
            in_tensor_name = sess.get_inputs()[0].name
            out_tensor_name = sess.get_outputs()[0].name

            input_img = read_image(img_path)
            if input_img is None:
                print(herror+"Image '"+img_path+"' could not be loaded!")
                return None
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            im_h, im_w, _ = input_img.shape

            imask = leaf_masking.process(input_img, MASKING_RL_TH)
            if imask is None:
                print(hwarning+"No sample found in: '"+str(img_path)+"'!")
                return None
            
            step = 1 # Assume step is always 1:1 resolution (no sub-image overlapping)
            step = math.floor(SUBIMAGE_WIDTH / step)

            # obtain remaining pixels at the last iteration
            offsetX = im_w % step
            offsetY = im_h % step

            n_xsteps =  math.floor(im_w / step)
            n_ysteps = math.floor(im_h / step)
            result_map = np.zeros((n_ysteps, n_xsteps), dtype=np.float32)

            xi = int(offsetX / 2)
            yi = int(offsetY / 2)

            #last_l_ind = n_xsteps * n_ysteps
            #sub_images_ind = [(0,0)] * last_l_ind # I dont need this..

            # TODO: Line 353 matlab
            #print("Y steps=", n_ysteps)
            #print("X steps=", n_xsteps)
            #print("Xi=", xi)
            #print("Yi=", yi)
            #print(last_l_ind)

            save_subimg = 0
            for i in range(n_ysteps):
                for j in range(n_xsteps):
                    # Compute current sub-image indexes
                    x = xi + (SUBIMAGE_WIDTH*j)
                    y = yi + (SUBIMAGE_HEIGHT*i)

                    # Crop mask subimage
                    subimg_mask = imask[y:y+SUBIMAGE_HEIGHT, x:x+SUBIMAGE_WIDTH]
                    
                    # If sub-image is not focused, insert NaN score, and skip
                    if not isOnFocus(subimg_mask):
                        result_map[i,j] = np.nan
                        continue
                    
                    # Crop sub-image and classify
                    subimg = input_img[y:y+SUBIMAGE_HEIGHT, x:x+SUBIMAGE_WIDTH]

                    #if save_subimg < 10:
                    #    cv2.imwrite("subimage_"+str(save_subimg)+".png",cv2.cvtColor(subimg, cv2.COLOR_RGB2BGR))
                    #    save_subimg+=1

                    # CNN pre-processing
                    subimg = subimg.astype(np.float32)
                    #subimg /= 255. # NOTE: Normalization layer inside the original CNNs!!!!!
                    subimg = np.expand_dims(subimg, axis=0)
                    subimg = np.transpose(subimg, (0,3,1,2)) # NHWC to NCHW (ONNX)
                    pred = sess.run([out_tensor_name], {in_tensor_name: subimg})[0][0]

                    # WARNING: TO BE CONFIRMED FOR EACH CNN -> pred[0] Infected, pred[1] Clear
                    # Use analyzeNetwork() func in MATLAB to check this
                    #print(pred)
                    result_map[i,j] = pred[out_pos]

                    #if save_subimg < 10:
                    #    print(pred)

                    #class_id = np.argmax(pred)
                    #if class_id == 0: # Infected
                    #    result_map[i,j] = pred[0]
                    #else:             # Clear
                    #    result_map[i,j] = pred[1]

            return result_map

        total_images = expdata["nimages"]
        progress[0] = 0.0
        samples_done = 0
        sample_times = []

        date_idx = 0
        tray_idx = 0
        samp_idx = 0
        expdir = Path(expdata["path"])

        ort_sess = ort.InferenceSession(model_path, providers=[GPU_BACKEND])
        # Should I get input tensor shape?

        # NOTE: TEMPORARY SOLUTION!!!
        # Find out which score in output tensor is the infected label!
        # Run a black image and take lowest prob as "Infected"
        im_black = np.zeros([1,3,224,224],dtype=np.float32)
        pred = ort_sess.run([ort_sess.get_outputs()[0].name], {ort_sess.get_inputs()[0].name: im_black})[0][0]
        infected_prob_idx = np.argmin(pred)

        for date in expdata["samples"]:
            for tray in expdata["samples"][date]:
                for sample in expdata["samples"][date][tray]:
                    
                    if stop[0]:
                        stop[0] = False
                        return {}
            
                    if sample is None:
                        continue

                    img_path = expdir / date / tray / sample
                    sample_idx = int(sample.split('-')[0]) - 1
                    #print(img_path)

                    t_start = time.time()
                    # Store result tuple of (sample_id_str, score_map)
                    expdata["results"][date][tray][sample_idx] = (os.path.splitext(sample)[0], compute_sample(ort_sess, img_path, infected_prob_idx))
                    t_end = time.time()
                    #print("Sample processing time:",t_end-t_start)
                    sample_times.append(t_end-t_start)

                    samples_done += 1

                    etc = statistics.mean(sample_times) * (total_images-samples_done)
                    tsec = int(etc % 60)
                    tmin = math.floor(etc / 60) % 60
                    thour = math.floor(etc / 3600)
                    etc_str[0] = '{:02d}'.format(thour) + ':' + '{:02d}'.format(tmin) + ':' + '{:02d}'.format(tsec)
                    progress[0] = float(samples_done) / float(total_images)

        return expdata["results"]
    

        

    def _update_progress(self):
        if self._tprocess.is_alive():
            # Update GUI
            self._pbar['value'] = self._progress[0]
            self._t_etc.set(f"{int(self._progress[0]*100)}% - ETC {self._etc[0]}")
            self._win.after(GUI_UPDATE_FREQ, self._update_progress)
        else:
            self._pbar['value'] = 1.0 # This does go here, what happends when it's interrupted?
            self._t_etc.set(f"100% - ETC 00:00:00")
            self._bt_start["state"] = tk.NORMAL
            self._bt_open["state"] = tk.NORMAL
            self._bt_stop["state"] = tk.DISABLED

            if len(self._tprocess.result):
                print("COMPLETED!")
                self._expdata["results"] = self._tprocess.result
                self._save_results()
            else:
                print("Cancelled!!")
            self._tprocess = None

    def _save_results(self):
        # Write results to msgpack file
        out_path = Path(self._expdata["path"]) / "results.msgpack"
        with open(out_path, "wb") as outfile:
            packed = msgpack.packb(self._expdata["results"])
            outfile.write(packed)

        # Read msgpack file
        #with open("data.msgpack", "rb") as data_file:
        #    byte_data = data_file.read()
        #data_loaded = msgpack.unpackb(byte_data)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackbird Samples Analyzer")
    parser.add_argument(
        "model",
        metavar="<ONNX_MODEL>",
        help="Path to ONNX classification model used for the analysis",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(herror + " Specified model path " + args.model + " does not exist!")
        exit()

    if os.path.splitext(args.model)[-1] != ".onnx":
        print(herror + " The specified model " + args.model + " is not in ONNX format!")
        exit()

    # Check if there is a GPU in the system
    # TODO ?

    win = tk.Tk()
    if os.path.exists("res\\icon.ico"):
        win.iconbitmap("res\\icon.ico")
    win.title("Blackbird Samples Analyzer")
    win.resizable(False, False)

    app = CNNAnalyzerGUI(win, args.model)
    win.mainloop()