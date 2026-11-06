import cv2
import numpy as np
import os
from PIL import Image


def focus_map(img, wsize=50, wstep=6):
    """
        Sharpness (focus) metric of the image using a sliding window: the maximum over the
        color channels of the standard deviation of each wsize x wsize window, with windows
        every wstep pixels. Window sums come from integral images instead of a Python loop.

    Args:
        img:        A RGB image as a Numpy array
    Return:
        fmat:       Focus metric map, one value per window
    """
    height, width, _ = img.shape

    # res_offset is equal to the number of wstep that can fit into wsize. This value will be used to calculate the offset of the resulting image.
    res_offset = int(wsize / wstep)
    aux_height = int((height / wstep) - res_offset)
    aux_width = int((width / wstep) - res_offset)

    # Top-left corner of every window
    ys = (np.arange(aux_height) * wstep)[:, None]
    xs = np.arange(aux_width) * wstep
    n = wsize * wsize

    def window_sum(integral):
        return integral[ys + wsize, xs + wsize] - integral[ys, xs + wsize] - integral[ys + wsize, xs] + integral[ys, xs]

    fmat = np.zeros((aux_height, aux_width), dtype=np.float64)
    for c in range(img.shape[2]):
        s, sq = cv2.integral2(np.ascontiguousarray(img[:, :, c]), sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
        mean = window_sum(s) / n
        stddev = np.sqrt(np.maximum(window_sum(sq) / n - mean * mean, 0))   # Same as cv2.meanStdDev
        np.maximum(fmat, stddev, out=fmat)
    return fmat


def leaf_mask(img, rel_th):
    """
        Fill holes in a binary image using cv2
        # Relative thresholding (0 < rth < 1)

    Args:
        img:        A RGB format image with PIL format (or Numpy array)
        rel_th:     Relative threshold value
    Return:
        im_out:     Foreground mask with Numpy array format (0/255, same size as img),
                    or None if no leaf is found
    """
    img = np.asarray(img)
    height, width, _ = img.shape
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Get sharpness/focused of the image by calculating the standard deviation
    fmat = focus_map(img)

    _, max_val, _, _ = cv2.minMaxLoc(fmat)
    # 1st Threshold
    _, imbin = cv2.threshold(fmat, max_val * rel_th,
                             255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Calculate the mean between mask and thresholded mask
    fv = cv2.mean(fmat, imbin)
    th = fv[0]
    # 2nd Threshold
    _, imbin = cv2.threshold(fmat, th, 255, cv2.THRESH_BINARY_INV)
    imbin = imbin.astype('uint8')
    # Bitwise not
    imbin1 = cv2.bitwise_not(imbin)
    imbin1 = imfill(imbin1)
    # Erode
    imbin1 = cv2.erode(imbin1, kernel=kernel, anchor=(-1, -1), iterations=10)

    # Add margins for erosion
    imbin1[[0, -1], :] = 0
    imbin1[:, [0, -1]] = 0
    imbin1 = cv2.erode(imbin1, kernel=kernel, anchor=(-1, -1), iterations=4)
    imbin1 = imfill(imbin1)

    h, w = imbin1.shape
    # Mask validation
    # The stats var is [num_of_connected_component, 5 (left, top, width, height, area)]
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(
        imbin1, connectivity=8, ltype=cv2.CV_16U)

    # No leaf pixels found
    if retval < 2:
        print('No leaf pixels found. No mask.')
        return None

    # Leaf area so small, skip
    big_l = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big_l][4] < (h * w * 0.15):
        print('Leaf area so small! No mask.')
        return None

    # Delete other objects, keep the biggest
    imbin1 = np.where(labels == big_l, 255, 0).astype(np.uint8)

    imbin1 = cv2.erode(imbin1, kernel=kernel, anchor=(-1, -1), iterations=10)
    imbin1 = cv2.resize(imbin1, dsize=(width, height), fx=0, fy=0,
                        interpolation=cv2.INTER_NEAREST)

    return imbin1


def imfill(inmask):
    """
        Do floodFill operation on the input mask
    """
    flood_mask = inmask.copy()
    h, w = flood_mask.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    for col in range(w):
        if flood_mask[0, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, 0),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[h - 1, col] == 0:
            cv2.floodFill(flood_mask, mask, (col, h - 1),
                          newVal=255, loDiff=10, upDiff=10)

    for row in range(h):
        if flood_mask[row, 0] == 0:
            cv2.floodFill(flood_mask, mask, (0, row),
                          newVal=255, loDiff=10, upDiff=10)
        if flood_mask[row, w - 1] == 0:
            cv2.floodFill(flood_mask, mask, (w - 1, row),
                          newVal=255, loDiff=10, upDiff=10)

    # Holes are the pixels the flood from the borders could not reach
    out_mask = inmask.copy()
    out_mask[flood_mask == 0] = 255

    return out_mask


def on_focus(imask, th=0.7):
    """
        Check image patches focused or not
    Args:
        imask:  Image patches' mask
    """
    mask_ratio = np.mean(imask)
    if mask_ratio > th:
        focused = True
    else:
        focused = False

    return focused

# if you are directly running this script, then you need to change timestamp, tray, directory below. If this script is being imported as a module, then you do not need to alter the parameters below. 
if __name__ == '__main__':

    from PIL import Image
    from matplotlib import pyplot as plt

    plt.rc('font', size=12)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('figure', titlesize=12)

    timestamp = '12-6-2022_5dpi'
    tray = '1'
    directory = f'/Users/michele.wiseman/Desktop/blackbird_ml/data/{timestamp}/{tray}'
    
    if os.path.exists(directory):
        print("The path exists")
    else:
        print("The path does not exist")
        
    if not os.path.isdir(directory):
        os.makedirs(directory)
        
    abs_path = os.path.abspath(directory)
    print(abs_path)

    

    for file_name in os.listdir(directory):
        if file_name.endswith('.png'):
            file_path = os.path.join(directory, file_name)
            img = Image.open(file_path)
            imask = leaf_mask(img, rel_th=0.15)
            img_arr = np.asarray(img)
            imask_arr = np.expand_dims(imask, axis=2)
            imask_arr = np.repeat(imask_arr, repeats=3, axis=2)
            masked_img_arr = img_arr * imask_arr
            img_arr_copy = img_arr.copy()
            img_arr_copy[imask == 0] = 0
            plt_fig, plt_axis = plt.subplots(figsize=(10, 6))
            plt_axis.imshow(img_arr)
            plt_axis.set_title('Raw leaf disk image')
            plt_axis.axis('off')
            plt_fig.savefig(f'{timestamp}_{file_name}.png', dpi=300)
            plt_fig, plt_axis = plt.subplots(figsize=(10, 6))
            plt_axis.imshow(img_arr_copy)
            plt_axis.set_title('Masked leaf disk image')
            plt_axis.axis('off')
            imask = Image.fromarray(imask)
            imask.save(f'/Users/michele.wiseman/Desktop/blackbird_ml/data/{timestamp}/{tray}/{file_name}_masked.png')

