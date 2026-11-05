"""
    Collects the analyzer results of many experiments into a single CSV table, with one row per
    sample image and the per-model metrics side by side. When there are several models (e.g.
    SporeNet3 for sporulation and Hyphal2019Net2 for hyphae), it also computes a combined
    severity from the union of the infected sub-images of all of them.
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import msgpack
import msgpack_numpy as mnp
mnp.patch()

# Useful colored strings
hwarning = "\033[93m[WARNING]\033[0m: "
herror = "\033[91m[ERROR]\033[0m: "

RESULTS_PREFIX = "results_"
RESULTS_EXT = ".msgpack"


def find_experiments(paths):
    """
        Returns {experiment_dir: {model_name: results_path}} for the given experiment or repository
        folders. A folder with results_<MODEL>.msgpack files is an experiment, otherwise its
        sub-folders are searched.
    """
    def results_files(folder):
        return {f.name[len(RESULTS_PREFIX):-len(RESULTS_EXT)]: Path(f.path) for f in os.scandir(folder)
                if f.is_file() and f.name.startswith(RESULTS_PREFIX) and f.name.endswith(RESULTS_EXT)}

    experiments = {}
    for path in paths:
        found = results_files(path)
        if found:
            experiments[Path(path)] = found
            continue
        for sub in sorted(f.path for f in os.scandir(path) if f.is_dir() and not f.name.startswith(".")):
            found = results_files(sub)
            if found:
                experiments[Path(sub)] = found
            else:
                print(hwarning + f"No results files found in '{sub}', skipping it.")
    return experiments


def load_results(path):
    with open(path, "rb") as data_file:
        return msgpack.unpackb(data_file.read())


def counts(infected, clear, analyzed):
    # Same definition as the thresholder: % infected = INF / (INF + CLR)
    n_inf = int(np.count_nonzero(infected))
    n_clr = int(np.count_nonzero(clear))
    pct = round(100 * n_inf / (n_inf + n_clr), 2) if (n_inf + n_clr) else ""
    return [int(analyzed), n_inf, n_clr, pct]


def model_metrics(score_map, low, high):
    # [analyzed, infected, clear, pct_infected, mean_score] of one sample for one model
    if score_map is None:   # No leaf found in the image
        return [0, 0, 0, "", ""]
    valid = ~np.isnan(score_map)
    scores = score_map[valid]
    mean = round(float(scores.mean()), 4) if scores.size else ""
    return counts(scores >= high, scores < low, scores.size) + [mean]


def union_metrics(score_maps, low, high):
    """
        Combined severity of several models on the same sample. A sub-image is infected if any model
        rates it as infected (union of the infected areas), and clear only if all the models rate it
        as clear. All models score the same sub-images since they share the image leaf mask.
    """
    if any(m is None for m in score_maps):
        if all(m is None for m in score_maps):
            return [0, 0, 0, ""]
        return ["", "", "", ""]     # Inconsistent results (e.g. models run on different images)
    if len({m.shape for m in score_maps}) != 1:
        return ["", "", "", ""]
    stack = np.stack(score_maps)                    # (n_models, rows, cols)
    valid = ~np.isnan(stack).any(axis=0)
    stack = stack[:, valid]                         # (n_models, n_analyzed)
    return counts((stack >= high).any(axis=0), (stack < low).all(axis=0), valid.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackbird Results Summarizer")
    parser.add_argument(
        "paths",
        metavar="<DIR>",
        nargs="+",
        help="Repository folder(s) containing experiment folders, or experiment folder(s), "
             "with results_<MODEL>.msgpack files written by the analyzer.",
    )
    parser.add_argument(
        "-o", "--out",
        metavar="<OUT_CSV>",
        default="summary.csv",
        help="Path of the resulting CSV file (default: summary.csv)",
    )
    parser.add_argument(
        "-m", "--models",
        metavar="<MODEL_NAME>",
        nargs="+",
        help="Models to include, by name (e.g. SporeNet3 Hyphal2019Net2). Default: all models found",
    )
    parser.add_argument(
        "-u", "--union",
        metavar="<MODEL_NAME>",
        nargs="+",
        help="Models combined in the union severity. Default: all included models, if there are two or more",
    )
    parser.add_argument(
        "-lo", "--low",
        metavar="<LOW_TH>",
        type=float,
        default=0.5,
        help="Sub-images with a score below this are clear (default: 0.5)",
    )
    parser.add_argument(
        "-hi", "--high",
        metavar="<HIGH_TH>",
        type=float,
        default=0.5,
        help="Sub-images with a score equal or above this are infected (default: 0.5)",
    )
    args = parser.parse_args()

    if not 0 <= args.low <= args.high <= 1:
        print(herror + " Thresholds must satisfy 0 <= low <= high <= 1!")
        exit(1)

    for path in args.paths:
        if not os.path.isdir(path):
            print(herror + " Folder '" + path + "' does not exist!")
            exit(1)

    experiments = find_experiments(args.paths)
    if not experiments:
        print(herror + " No results_<MODEL>.msgpack files found.")
        exit(1)

    found_models = sorted({m for exp in experiments.values() for m in exp})
    models = args.models or found_models
    missing = [m for m in models if m not in found_models]
    if missing:
        print(herror + f" No results found for model(s): {', '.join(missing)}. Found: {', '.join(found_models)}")
        exit(1)

    union = args.union or (models if len(models) > 1 else [])
    if any(m not in models for m in union):
        print(herror + " Union models must be among the included models.")
        exit(1)
    if len(union) == 1:
        print(herror + " The union needs at least two models.")
        exit(1)

    header = ["experiment", "timepoint", "date", "tray", "sample_number", "sample_id"]
    for m in models:
        header += [f"{m}_analyzed", f"{m}_infected", f"{m}_clear", f"{m}_pct_infected", f"{m}_mean_score"]
    if union:
        header += ["union_analyzed", "union_infected", "union_clear", "union_pct_infected"]

    n_rows = 0
    with open(args.out, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)

        for expdir, model_files in experiments.items():
            results = {m: load_results(model_files[m]) for m in models if m in model_files}
            not_run = [m for m in models if m not in results]
            if not_run:
                print(hwarning + f"'{expdir.name}' has no results for {', '.join(not_run)}, those columns are left empty.")

            # All (timepoint, tray, sample) keys found in any model, timepoints in date order
            samples = {}
            for model_results in results.values():
                for timepoint, trays in model_results.items():
                    for tray, tray_samples in trays.items():
                        for idx, sample in enumerate(tray_samples):
                            if sample is not None:
                                samples.setdefault((timepoint, tray, idx), sample[0])

            def sort_key(key):
                date = datetime.strptime(key[0].split("_")[0], "%m-%d-%Y")
                return (date, key[0], key[1], key[2])

            for (timepoint, tray, idx) in sorted(samples, key=sort_key):
                date = datetime.strptime(timepoint.split("_")[0], "%m-%d-%Y").strftime("%Y-%m-%d")
                row = [expdir.name, timepoint, date, tray, idx + 1, samples[(timepoint, tray, idx)]]

                maps = {}
                for m in models:
                    sample = results.get(m, {}).get(timepoint, {}).get(tray, [None]*(idx+1))[idx]
                    if sample is None:
                        row += [""] * 5
                    else:
                        maps[m] = sample[1]
                        row += model_metrics(sample[1], args.low, args.high)

                if union:
                    if all(m in maps for m in union):
                        row += union_metrics([maps[m] for m in union], args.low, args.high)
                    else:
                        row += [""] * 4

                writer.writerow(row)
                n_rows += 1

    print(f"{n_rows} samples from {len(experiments)} experiments ({', '.join(models)}"
          + (f"; union of {' + '.join(union)}" if union else "") + f") saved to: {os.path.abspath(args.out)}")
