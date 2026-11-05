"""
    Compares the compiled leaf masking (cpp_functions, Windows only) with the pure Python
    one (leaf_mask.py) on full sample images. Run it from the repository folder on Windows:

        python test\\compare_leaf_masks.py test\\123-DVIT2923_WEOL1607.png test\\309-DVIT2301_WEOW0706.png
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cpp_functions.leaf_masking as cpp_masking
from leaf_mask import leaf_mask
from analyzer import read_image, MASKING_RL_TH, SUBIMAGE_WIDTH, SUBIMAGE_HEIGHT


def focus_grid(mask, im_h, im_w):
    # Same on-focus sub-image grid as the analyzer
    ny, nx = im_h // SUBIMAGE_HEIGHT, im_w // SUBIMAGE_WIDTH
    yi, xi = (im_h % SUBIMAGE_HEIGHT) // 2, (im_w % SUBIMAGE_WIDTH) // 2
    grid = mask[yi:yi + ny*SUBIMAGE_HEIGHT, xi:xi + nx*SUBIMAGE_WIDTH].reshape(ny, SUBIMAGE_HEIGHT, nx, SUBIMAGE_WIDTH)
    return grid.mean(axis=(1, 3)) / 255 > 0.7


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare compiled vs Python leaf masking")
    parser.add_argument("images", metavar="<IMAGE>", nargs="+", help="Full sample image(s).")
    args = parser.parse_args()

    for img_path in args.images:
        img = cv2.cvtColor(read_image(img_path), cv2.COLOR_BGR2RGB)
        im_h, im_w, _ = img.shape

        t = time.time(); m_cpp = cpp_masking.process(img, MASKING_RL_TH); t_cpp = time.time() - t
        t = time.time(); m_py = leaf_mask(img, MASKING_RL_TH); t_py = time.time() - t

        print(f"{os.path.basename(img_path)}: compiled {t_cpp:.2f} s, Python {t_py:.2f} s")
        if m_cpp is None or m_py is None:
            print(f"  No mask -> compiled: {m_cpp is None}, Python: {m_py is None}")
            continue
        if m_cpp.ndim == 3:
            m_cpp = m_cpp[:, :, 0]
        print(f"  Pixels differing: {int(((m_cpp > 0) != (m_py > 0)).sum())} of {m_py.size}")
        f_cpp, f_py = focus_grid(m_cpp, im_h, im_w), focus_grid(m_py, im_h, im_w)
        print(f"  On-focus sub-images: compiled {int(f_cpp.sum())}, Python {int(f_py.sum())}, differing {int((f_cpp != f_py).sum())}")
