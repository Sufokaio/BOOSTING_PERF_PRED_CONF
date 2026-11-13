# HINNPerf prediction time without retraining.
import os

for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_v] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import glob
import random
import sys
import time
import warnings
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from experiments.val_scores import _loader
from evaluation.metrics import compute_all

warnings.filterwarnings("ignore")

BLOCKS = [2, 3, 4]
LAYERS = [2, 3, 4]
N_RUNS = 5
N_REPEAT = 5
COLUMNS = ["dataset", "sample_size", "run", "num_block", "num_layer_pb", "n_test",
           "fit_s", "pred_first_s", "pred_median_s", "MRE"]


def _single_thread_session(self):
    import tensorflow.compat.v1 as tf
    conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    self.sess = tf.Session(config=conf)
    self.sess.run(tf.global_variables_initializer())


def measure(X_train, y_train, X_test, y_test, num_block, num_layer_pb):
    from HINNPERF.HINNPerf_data_preproc import DataPreproc
    from HINNPERF.HINNPerf_models import MLPHierarchicalModel
    from HINNPERF.HINNPerf_model_runner import ModelRunner

    MLPHierarchicalModel.init_session = _single_thread_session
    whole = np.vstack([np.column_stack([X_train, y_train]), np.column_stack([X_test, np.zeros(len(X_test))])])
    data = DataPreproc(whole, np.arange(len(X_train)), np.arange(len(X_train), len(whole)))
    runner = ModelRunner(data, MLPHierarchicalModel)
    conf = dict(input_dim=data.config_num, num_neuron=128, num_block=num_block, num_layer_pb=num_layer_pb,
                lamda=0.001, linear=False, gnorm=True, lr=0.001, decay=None, verbose=False)

    X_tr, Y_tr, X_te, _, mean_Y, std_Y, _ = data.get_train_test_samples(conf["gnorm"])
    model = MLPHierarchicalModel(conf)
    model.build_train()
    t0 = time.perf_counter()
    lr = conf["lr"]
    decay = lr / 1000
    for epoch in range(1, runner.max_epoch):
        model.sess.run([model.train_op, model.loss, model.output], {model.X: X_tr, model.Y: Y_tr, model.lr: lr})
        lr = lr * 1 / (1 + decay * epoch)
    fit_s = time.perf_counter() - t0

    def predict():
        out = model.sess.run([model.output], {model.X: X_te})
        return np.asarray(out * (std_Y + (std_Y == 0) * .001) + mean_Y).ravel()

    t0 = time.perf_counter()
    pred = predict()
    first = time.perf_counter() - t0
    reps = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        predict()
        reps.append(time.perf_counter() - t0)
    model.finalize()
    return fit_s, first, float(np.median(reps)), compute_all(y_test, pred)["MRE"]


def part_path(out_dir, dataset, size, run, b, l):
    return os.path.join(out_dir, "parts", dataset, str(size), f"run{run:02d}_b{b}_l{l}.csv")


def run_task(task):
    dataset, size, run, b, l, out_dir = task
    path = part_path(out_dir, dataset, size, run, b, l)
    if os.path.exists(path):
        return task, 0.0, "skipped"
    t0 = time.perf_counter()
    seed = cfg.RANDOM_SEED_BASE + run
    random.seed(seed)
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = _loader(dataset).sample(size, seed)
    fit_s, first, med, mre = measure(X_tr, y_tr, X_te, y_te, b, l)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = [(dataset, size, run, b, l, len(X_te), fit_s, first, med, mre)]
    tmp = path + ".tmp"
    pd.DataFrame(row, columns=COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return task, time.perf_counter() - t0, "done"


def combine(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "parts", "*", "*", "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(out_dir, "hinnperf_pred_time.csv"), index=False)
    print(f"combined {len(files)} files, {len(df)} rows -> {os.path.join(out_dir, 'hinnperf_pred_time.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/val/hinnperf_time")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    if args.combine:
        combine(args.out)
        return

    datasets = args.datasets or list(cfg.DATASETS)
    shapes = sorted(product(BLOCKS, LAYERS), key=lambda s: (s not in [(2, 2), (4, 4)], s))
    tasks = [(d, s, r, b, l, args.out) for (b, l) in shapes for d in datasets
             for s in cfg.SAMPLE_SIZES[d] for r in range(args.runs)]
    os.makedirs(args.out, exist_ok=True)
    t_start = time.perf_counter()
    with Pool(args.processes) as pool, open(os.path.join(args.out, "progress.log"), "a") as log:
        for n, ((d, s, r, b, l, _), elapsed, status) in enumerate(pool.imap_unordered(run_task, tasks), start=1):
            line = (f"{time.strftime('%H:%M:%S')} {n}/{len(tasks)} {status} {d} {s} run{r:02d} b{b} l{l} "
                    f"{elapsed:.1f}s (wall {time.perf_counter() - t_start:.0f}s)")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    main()
