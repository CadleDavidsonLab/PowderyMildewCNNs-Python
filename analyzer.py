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
from threading import Thread
import multiprocessing as mp
import queue
import traceback
import math

import onnxruntime as ort
import numpy as np
import cv2
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:     # Headless systems (e.g. HPC nodes) may not have Tk, only --exp mode works there
    tk = None
import msgpack
import msgpack_numpy as mnp
mnp.patch()



try:
    # Compiled leaf masking (Windows, Python 3.11 only)
    from cpp_functions.leaf_masking import process as leaf_masking
except ImportError:
    # Pure Python implementation of the same algorithm, works on any platform
    from leaf_mask import leaf_mask as leaf_masking

# Useful colored strings
hwarning = "\033[93m[WARNING]\033[0m: "
herror = "\033[91m[ERROR]\033[0m: "

CPU_BACKEND = "CPUExecutionProvider"
CUDA_BACKEND = "CUDAExecutionProvider"     # NVIDIA GPUs, Linux/Windows (onnxruntime-gpu package)
DML_BACKEND = "DmlExecutionProvider"       # Any DirectX 12 GPU, Windows only (onnxruntime-directml package)

N_SAMPLES_X_TRAY = 351

MASKING_RL_TH = 0.2         # Relative threshold used for leaf masking algorithm

SUBIMAGE_HEIGHT = 224
SUBIMAGE_WIDTH = 224

GUI_UPDATE_FREQ = 1000 # ms

RESULTS_FILENAME = "results.msgpack"

DEFAULT_BATCH_SIZE = 64     # Sub-images classified per inference run
DEFAULT_LOADERS = 2         # Image loading/masking threads per device


# ---------------------------------------------------------------------------
# Inference functions. They are module-level so that they can be run inside
# the worker processes (one per device) launched by the analysis.
# ---------------------------------------------------------------------------

def read_image(img_path):
    """
        NOTE: This function is implemented to read images from an UTF-8 path string
    """
    with open(img_path, "rb") as stream:
        numpyarray = np.frombuffer(stream.read(), dtype=np.uint8)
    return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED) # Returns BGR image


def load_model(model_path):
    """
        Returns the model (path or serialized bytes) and whether it accepts batches of sub-images.
        Models exported from MATLAB with 'BatchSize', 1 have a fixed batch dimension, which is
        made dynamic here so that many sub-images can be classified in a single run.
    """
    sess = ort.InferenceSession(model_path, providers=[CPU_BACKEND])
    if not isinstance(sess.get_inputs()[0].shape[0], int):
        return model_path, True     # Batch dimension is already dynamic
    del sess

    try:
        import onnx
    except ImportError:
        print(hwarning+"'onnx' package is not installed, sub-images will be classified one by one (slow).")
        return model_path, False

    model = onnx.load(model_path)
    for t in list(model.graph.input) + list(model.graph.output):
        t.type.tensor_type.shape.dim[0].dim_param = "N"
    del model.graph.value_info[:]   # Drop intermediate shapes that were fixed to batch size 1
    return model.SerializeToString(), True


def gpu_backend():
    """
        Returns the GPU execution provider of the installed onnxruntime package, or None.
    """
    available = ort.get_available_providers()
    for backend in (CUDA_BACKEND, DML_BACKEND):
        if backend in available:
            return backend
    return None


def create_session(model, device_id):
    """
        Creates an inference session on the given GPU device id, or on the CPU if None.
    """
    if device_id is None:
        return ort.InferenceSession(model, providers=[CPU_BACKEND])

    backend = gpu_backend()
    if backend is None:
        raise RuntimeError("The installed onnxruntime has no GPU support. Install onnxruntime-gpu (NVIDIA, Linux) "
                           "or onnxruntime-directml (Windows), or run with --cpu.")

    so = ort.SessionOptions()
    if backend == DML_BACKEND:
        # Required by the DirectML execution provider
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(model, sess_options=so, providers=[(backend, {"device_id": device_id})])

    # onnxruntime silently falls back to the CPU if the GPU provider fails to start
    if sess.get_providers()[0] != backend:
        raise RuntimeError(f"{backend} could not start on GPU {device_id}. Check the GPU id and drivers "
                           "(and the CUDA/cuDNN versions required by onnxruntime-gpu).")
    return sess


def prepare_sample(img_path):
    """
        Loads a sample image, masks the leaf and crops all its on-focus sub-images at once.
        Returns (subimgs, focus): subimgs is a (K, 3, H, W) uint8 array with the K on-focus
        sub-images in row-major order, and focus is the (n_ysteps, n_xsteps) boolean grid.
    """
    input_img = read_image(img_path)
    if input_img is None:
        print(herror+"Image '"+str(img_path)+"' could not be loaded!")
        return None
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    im_h, im_w, _ = input_img.shape

    imask = leaf_masking(input_img, MASKING_RL_TH)
    if imask is None:
        print(hwarning+"No sample found in: '"+str(img_path)+"'!")
        return None
    if imask.ndim == 3:
        imask = imask[:, :, 0]

    # Sub-images do not overlap (1:1 step), remaining pixels are split between both borders
    n_xsteps = im_w // SUBIMAGE_WIDTH
    n_ysteps = im_h // SUBIMAGE_HEIGHT
    xi = (im_w % SUBIMAGE_WIDTH) // 2
    yi = (im_h % SUBIMAGE_HEIGHT) // 2
    rows = slice(yi, yi + n_ysteps*SUBIMAGE_HEIGHT)
    cols = slice(xi, xi + n_xsteps*SUBIMAGE_WIDTH)

    # A sub-image is on focus if more than 70% of it is masked as leaf
    mask_grid = imask[rows, cols].reshape(n_ysteps, SUBIMAGE_HEIGHT, n_xsteps, SUBIMAGE_WIDTH)
    focus = mask_grid.mean(axis=(1, 3)) / 255 > 0.7

    # (ny, H, nx, W, C) -> (ny, nx, C, H, W) grid of NCHW sub-images (a view, no copy yet)
    img_grid = input_img[rows, cols].reshape(n_ysteps, SUBIMAGE_HEIGHT, n_xsteps, SUBIMAGE_WIDTH, 3)
    subimgs = img_grid.transpose(0, 2, 4, 1, 3)[focus]  # Copies only the on-focus sub-images
    return subimgs, focus


def classify_subimages(sess, subimgs, focus, out_pos, batch_size):
    """
        Classifies the sub-images in batches. Returns the score map with NaN on the
        sub-images that are not on focus.
    """
    in_tensor_name = sess.get_inputs()[0].name
    out_tensor_name = sess.get_outputs()[0].name

    scores = np.empty(len(subimgs), dtype=np.float32)
    for s in range(0, len(subimgs), batch_size):
        # NOTE: Normalization layer inside the original CNNs, so no /255 here!
        batch = subimgs[s:s+batch_size].astype(np.float32)
        # WARNING: TO BE CONFIRMED FOR EACH CNN -> pred[0] Infected, pred[1] Clear
        scores[s:s+batch_size] = sess.run([out_tensor_name], {in_tensor_name: batch})[0][:, out_pos]

    result_map = np.full(focus.shape, np.nan, dtype=np.float32)
    result_map[focus] = scores
    return result_map


def inference_worker(model, device_id, batch_size, n_loaders, task_q, result_q):
    """
        Runs in its own process, one per device. Loader threads read and mask the next images
        (CPU work) while the main thread keeps the device busy classifying sub-images.
        Sends ("result", task, score_map) per sample, then ("done", device_id, None), or
        ("error", device_id, traceback_str) if anything fails.
    """
    try:
        sess = create_session(model, device_id)

        # NOTE: TEMPORARY SOLUTION!!!
        # Find out which score in output tensor is the infected label!
        # Run a black image and take lowest prob as "Infected"
        im_black = np.zeros([1, 3, SUBIMAGE_HEIGHT, SUBIMAGE_WIDTH], dtype=np.float32)
        pred = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: im_black})[0][0]
        infected_prob_idx = int(np.argmin(pred))

        # Bounded so loaders stay only a few images ahead (each image is ~100s of MB in memory)
        prepared_q = queue.Queue(maxsize=n_loaders)

        def loader():
            while True:
                task = task_q.get()
                if task is None:    # One stop sentinel per loader thread
                    break
                try:
                    prepared_q.put((task, prepare_sample(task[-1])))
                except Exception:
                    prepared_q.put((task, traceback.format_exc()))
            prepared_q.put(None)

        for _ in range(n_loaders):
            Thread(target=loader, daemon=True).start()

        finished_loaders = 0
        while finished_loaders < n_loaders:
            item = prepared_q.get()
            if item is None:
                finished_loaders += 1
                continue
            task, prepared = item
            if isinstance(prepared, str):
                raise RuntimeError("Failed processing '"+task[-1]+"':\n"+prepared)
            result_map = None if prepared is None else classify_subimages(sess, *prepared, infected_prob_idx, batch_size)
            result_q.put(("result", task, result_map))

        result_q.put(("done", device_id, None))
    except Exception:
        result_q.put(("error", device_id, traceback.format_exc()))


def get_expdata(expdir):
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


def run_analysis(expdata, model_path, devices, batch_size, n_loaders, etc_str, progress, stop, log_progress=False) -> dict:
    """
        Analyzes all the experiment images on the given devices (GPU ids, None = CPU). etc_str,
        progress and stop are single-item lists shared with the caller (e.g. the GUI thread).
        Returns the results dict, or {} if stopped or failed.
    """
    # Every sample image is a task. Tasks are shared by all devices through a queue, so
    # faster devices simply take more of them.
    expdir = Path(expdata["path"])
    tasks = []
    for date in expdata["samples"]:
        for tray in expdata["samples"][date]:
            for sample in expdata["samples"][date][tray]:
                if sample is None:
                    continue
                sample_idx = int(sample.split('-')[0]) - 1
                tasks.append((date, tray, sample_idx, sample, str(expdir / date / tray / sample)))

    total_images = len(tasks)
    progress[0] = 0.0
    samples_done = 0

    model, batchable = load_model(model_path)
    if not batchable:
        batch_size = 1

    # "spawn" is the only start method on Windows, set it explicitly to behave the same everywhere
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in range(len(devices) * n_loaders):
        task_q.put(None)

    dev_names = ["CPU" if d is None else "GPU "+str(d) for d in devices]
    print(f"Analyzing {total_images} images on {', '.join(dev_names)} (batch size {batch_size}, {n_loaders} loaders per device)")

    workers = [ctx.Process(target=inference_worker,
                           args=(model, d, batch_size, n_loaders, task_q, result_q),
                           daemon=True)
               for d in devices]
    for w in workers:
        w.start()

    t_start = time.time()
    finished_workers = 0
    try:
        while finished_workers < len(workers):
            if stop[0]:
                stop[0] = False
                return {}

            try:
                kind, info, result_map = result_q.get(timeout=0.5)
            except queue.Empty:
                if not any(w.is_alive() for w in workers):
                    print(herror+"Analysis workers exited unexpectedly!")
                    return {}
                continue

            if kind == "error":
                print(herror+"Analysis failed on "+("CPU" if info is None else "GPU "+str(info))+":\n"+result_map)
                return {}
            if kind == "done":
                finished_workers += 1
                continue

            # Store result tuple of (sample_id_str, score_map)
            date, tray, sample_idx, sample, _ = info
            expdata["results"][date][tray][sample_idx] = (os.path.splitext(sample)[0], result_map)
            samples_done += 1

            # Estimated time to completion from the overall throughput (samples run in parallel)
            etc = (time.time() - t_start) / samples_done * (total_images - samples_done)
            tsec = int(etc % 60)
            tmin = math.floor(etc / 60) % 60
            thour = math.floor(etc / 3600)
            etc_str[0] = '{:02d}'.format(thour) + ':' + '{:02d}'.format(tmin) + ':' + '{:02d}'.format(tsec)
            progress[0] = float(samples_done) / float(total_images)
            if log_progress:
                print(f"[{samples_done}/{total_images}] {date}/{tray}/{sample} - ETC {etc_str[0]}", flush=True)
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join()

    print(f"Analysis time: {time.time() - t_start:.1f} s")
    return expdata["results"]


def save_results(expdata):
    # Write results to msgpack file
    out_path = Path(expdata["path"]) / RESULTS_FILENAME
    with open(out_path, "wb") as outfile:
        packed = msgpack.packb(expdata["results"])
        outfile.write(packed)
    return out_path


class ReturnableThread(Thread):
    # This class is a subclass of Thread that allows the thread to return a value.
    def __init__(self, target, args):
        Thread.__init__(self, args=args)
        self.target = target
        self.args = args
        self.result = None
    
    def run(self) -> None:
        self.result = self.target(*self.args)


class CNNAnalyzerGUI(tk.Frame if tk else object):
    def __init__(self,
                 master,
                 model_path,
                 devices=(0,),
                 batch_size=DEFAULT_BATCH_SIZE,
                 n_loaders=DEFAULT_LOADERS
    ):
        tk.Frame.__init__(self, master=master)

        # -------------- Internal Variables
        self._win = master
        self._model_path = model_path
        self._devices = list(devices)   # DirectML device ids, None = CPU
        self._batch_size = batch_size
        self._n_loaders = n_loaders
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
            if os.path.exists(Path(expdir)/RESULTS_FILENAME):
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

        self._expdata = get_expdata(expdir)

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
        
        self._tprocess = ReturnableThread(target=run_analysis, args=(self._expdata,
                                                                            self._model_path,
                                                                            self._devices,
                                                                            self._batch_size,
                                                                            self._n_loaders,
                                                                            self._etc,
                                                                            self._progress,
                                                                            self._interrupt))
        self._tprocess.start()

        self._win.after(GUI_UPDATE_FREQ, self._update_progress)
        pass


    def _stop_callback(self):
        self._interrupt[0] = True

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
                save_results(self._expdata)
            else:
                print("Cancelled!!")
            self._tprocess = None

if __name__ == "__main__":
    mp.freeze_support()    # Needed if packaged as an executable (worker processes)

    parser = argparse.ArgumentParser(description="Blackbird Samples Analyzer")
    parser.add_argument(
        "model",
        metavar="<ONNX_MODEL>",
        help="Path to ONNX classification model used for the analysis",
    )
    parser.add_argument(
        "-e", "--exp",
        metavar="<EXP_DIR>",
        nargs="+",
        help="Experiment folder(s) to analyze without the GUI (e.g. on a Linux HPC node). "
             "Results are written to <EXP_DIR>/" + RESULTS_FILENAME,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="With --exp, overwrite existing results instead of skipping the experiment",
    )
    parser.add_argument(
        "-g", "--gpus",
        metavar="<IDS>",
        default="0",
        help="Comma-separated GPU ids to run the analysis on, e.g. 0,1, or 'all' for every GPU in "
             "CUDA_VISIBLE_DEVICES (as set by Slurm) (default: 0)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the analysis on the CPU instead of the GPU(s)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        metavar="<N>",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Sub-images classified per inference run; lower it if the GPU runs out of memory (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "-l", "--loaders",
        metavar="<N>",
        type=int,
        default=DEFAULT_LOADERS,
        help=f"Image loading/masking threads per device (default: {DEFAULT_LOADERS})",
    )
    args = parser.parse_args()

    if args.cpu:
        devices = [None]
    elif args.gpus == "all":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            print(herror + " --gpus all needs CUDA_VISIBLE_DEVICES to be set (e.g. by Slurm), pass the GPU ids instead.")
            exit(1)
        # CUDA renumbers the visible GPUs from 0
        devices = list(range(len(visible.split(","))))
    else:
        try:
            devices = sorted({int(d) for d in args.gpus.split(",")})
        except ValueError:
            print(herror + " Invalid GPU ids '" + args.gpus + "', expected e.g. 0,1 or all")
            exit(1)

    if args.batch_size < 1 or args.loaders < 1:
        print(herror + " Batch size and number of loaders must be at least 1!")
        exit(1)

    if not os.path.exists(args.model):
        print(herror + " Specified model path " + args.model + " does not exist!")
        exit(1)

    if os.path.splitext(args.model)[-1] != ".onnx":
        print(herror + " The specified model " + args.model + " is not in ONNX format!")
        exit(1)

    if devices != [None] and gpu_backend() is None:
        print(herror + " The installed onnxruntime has no GPU support. Install onnxruntime-gpu (NVIDIA, Linux) "
                       "or onnxruntime-directml (Windows), or run with --cpu.")
        exit(1)

    # ---------- Headless mode ----------
    if args.exp:
        failed = []
        for expdir in args.exp:
            print(f"\n=== {expdir} ===")
            if os.path.exists(Path(expdir) / RESULTS_FILENAME) and not args.overwrite:
                print(hwarning + "Experiment already has results, skipping it (use --overwrite to redo it).")
                continue
            expdata = get_expdata(expdir)
            if expdata is None or expdata["nimages"] == 0:
                print(herror + "No sample images found in " + expdir)
                failed.append(expdir)
                continue
            results = run_analysis(expdata, args.model, devices, args.batch_size, args.loaders,
                                   ["00:00:00"], [0.0], [False], log_progress=True)
            if not results:
                failed.append(expdir)
                continue
            expdata["results"] = results
            print("Results saved to", save_results(expdata))

        if failed:
            print(herror + "Analysis failed for: " + ", ".join(failed))
            exit(1)
        exit(0)

    # ---------- GUI mode ----------
    if tk is None:
        print(herror + " Tkinter is not available on this system, use --exp to run without the GUI.")
        exit(1)

    win = tk.Tk()
    if os.path.exists("res\\icon.ico"):
        win.iconbitmap("res\\icon.ico")
    win.title("Blackbird Samples Analyzer")
    win.resizable(False, False)

    app = CNNAnalyzerGUI(win, args.model, devices, args.batch_size, args.loaders)
    win.mainloop()
