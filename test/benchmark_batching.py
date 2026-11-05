"""
    Times sub-image classification of full sample images for several batch sizes, to pick
    the --batch-size of the analyzer for a given GPU. Run it from the repository folder:

        python test\\benchmark_batching.py models\\SporeNet3.onnx test\\123-DVIT2923_WEOL1607.png --gpu 0
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analyzer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackbird analyzer batch size benchmark")
    parser.add_argument("model_path", metavar="<MODEL_PATH>", help="Path to model as a *.onnx file.")
    parser.add_argument("images", metavar="<IMAGE>", nargs="+", help="Full sample image(s) to analyze.")
    parser.add_argument("--gpu", type=int, default=0, help="DirectML GPU id (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Benchmark on the CPU instead")
    parser.add_argument("--batch-sizes", default="1,16,32,64,128,256",
                        help="Comma-separated batch sizes to test (default: 1,16,32,64,128,256)")
    args = parser.parse_args()

    model, batchable = analyzer.load_model(args.model_path)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")] if batchable else [1]
    sess = analyzer.create_session(model, None if args.cpu else args.gpu)
    print("Providers:", sess.get_providers())

    t_start = time.time()
    samples = [analyzer.prepare_sample(img) for img in args.images]
    samples = [s for s in samples if s is not None]
    n_subimgs = sum(len(s[0]) for s in samples)
    print(f"Loading + masking {len(samples)} images: {time.time() - t_start:.2f} s ({n_subimgs} on-focus sub-images)")

    for batch_size in batch_sizes:
        analyzer.classify_subimages(sess, samples[0][0][:batch_size], np.ones(min(batch_size, len(samples[0][0])), bool), 0, batch_size)  # Warm-up
        t_start = time.time()
        for subimgs, focus in samples:
            analyzer.classify_subimages(sess, subimgs, focus, 0, batch_size)
        t = time.time() - t_start
        print(f"Batch size {batch_size:4d}: {t:7.2f} s  ({n_subimgs / t:7.1f} sub-images/s)")
