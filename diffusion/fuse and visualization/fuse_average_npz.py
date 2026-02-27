#!/usr/bin/env python3
"""
Average two result npz files (e.g., bias and denoised) and save +avg output.
"""

import argparse
import os
import glob
import numpy as np


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return dict(data)


def squeeze_leading_one(x):
    x = np.array(x)
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    return x


def weighted_arrays(a, b, w):
    a = squeeze_leading_one(a)
    b = squeeze_leading_one(b)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return a
    return a[:n] * w + b[:n] * (1.0 - w)


def calculate_ate_eval(poses, poses_gt_full):
    poses = np.asarray(poses)
    poses_gt = np.asarray(poses_gt_full)[1:]
    n = min(poses.shape[0], poses_gt.shape[0])
    if n == 0:
        return float('nan')
    err = poses[:n] - poses_gt[:n]
    rmse = np.sqrt(np.mean(np.sum(err ** 2, axis=-1)))
    return float(rmse)


def calculate_rte_eval(poses, poses_gt_full, duration):
    poses = np.asarray(poses)
    poses_gt = np.asarray(poses_gt_full)[1:]
    if duration < 1:
        duration = 1
    n = poses.shape[0]
    if duration > n:
        duration = n
    if poses_gt.shape[0] < n:
        n = poses_gt.shape[0]
        poses = poses[:n]
        poses_gt = poses_gt[:n]
    if duration <= 1:
        dp = poses[1:] - poses[:-1]
        dp_gt = poses_gt[1:] - poses_gt[:-1]
    else:
        dp = poses[duration - 1:] - poses[:-duration + 1]
        dp_gt = poses_gt[duration - 1:] - poses_gt[:-duration + 1]
    rte = np.linalg.norm(dp - dp_gt, axis=-1)
    return float(np.mean(rte))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing npz files")
    parser.add_argument("--p1-suffix", default="+denoised.npz", help="Suffix for file 1")
    parser.add_argument("--p2-suffix", default="+bias.npz", help="Suffix for file 2")
    parser.add_argument("--out-suffix", default="+avg.npz", help="Suffix for output file")
    parser.add_argument("--rte-duration", type=int, default=100, help="Duration (frames) for RTE")
    parser.add_argument("--weight", type=float, default=0.5, help="Weight for file 1 (p1) when not searching")
    parser.add_argument("--search-weight", action="store_true", help="Search best weight on the folder")
    parser.add_argument("--w-min", type=float, default=0.0, help="Min weight for search")
    parser.add_argument("--w-max", type=float, default=1.0, help="Max weight for search")
    parser.add_argument("--w-step", type=float, default=0.05, help="Step for weight search")
    parser.add_argument("--alpha", type=float, default=1.0, help="RTE weight for combo loss")
    parser.add_argument("--save-best", action="store_true", help="Save outputs with best searched weight")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise ValueError(f"Directory not found: {args.dir}")

    p1_files = glob.glob(os.path.join(args.dir, f"*{args.p1_suffix}"))
    if not p1_files:
        print("No matching files found.")
        return

    def iter_pairs():
        for p1_path in p1_files:
            filename = os.path.basename(p1_path)
            prefix = filename[:-len(args.p1_suffix)]
            p2_path = os.path.join(args.dir, prefix + args.p2_suffix)
            if not os.path.exists(p2_path):
                print(f"Skip: {filename} (missing {args.p2_suffix})")
                continue
            yield prefix, p1_path, p2_path

    best_weight = args.weight
    if args.search_weight:
        w_min = args.w_min
        w_max = args.w_max
        w_step = args.w_step
        if w_step <= 0:
            raise ValueError("--w-step must be > 0")
        if w_max < w_min:
            raise ValueError("--w-max must be >= --w-min")

        best_score = float("inf")
        best_ate = float("inf")
        best_rte = float("inf")
        w = w_min
        while w <= w_max + 1e-12:
            ate_list = []
            rte_list = []
            for prefix, p1_path, p2_path in iter_pairs():
                d1 = load_npz(p1_path)
                d2 = load_npz(p2_path)
                poses1 = d1.get("poses", d1.get("P", None))
                poses2 = d2.get("poses", d2.get("P", None))
                if poses1 is None or poses2 is None:
                    continue
                poses_gt = d1.get("poses_gt", d1.get("Pgt", None))
                if poses_gt is None:
                    poses_gt = d2.get("poses_gt", None)
                if poses_gt is None:
                    continue
                poses_w = weighted_arrays(poses1, poses2, w)
                ate = calculate_ate_eval(poses_w, poses_gt)
                rte = calculate_rte_eval(poses_w, poses_gt, args.rte_duration)
                if not np.isnan(ate) and not np.isnan(rte):
                    ate_list.append(ate)
                    rte_list.append(rte)

            if ate_list and rte_list:
                mean_ate = float(np.mean(ate_list))
                mean_rte = float(np.mean(rte_list))
                score = mean_ate + args.alpha * mean_rte
                if score < best_score:
                    best_score = score
                    best_weight = w
                    best_ate = mean_ate
                    best_rte = mean_rte
            w += w_step

        print(
            f"Best weight: {best_weight:.4f} | mean ATE: {best_ate:.6f} | "
            f"mean RTE: {best_rte:.6f} | score: {best_score:.6f}"
        )
        if not args.save_best:
            return

    for prefix, p1_path, p2_path in iter_pairs():
        filename = os.path.basename(p1_path)

        d1 = load_npz(p1_path)
        d2 = load_npz(p2_path)

        poses1 = d1.get("poses", d1.get("P", None))
        poses2 = d2.get("poses", d2.get("P", None))
        vel1 = d1.get("vel", None)
        vel2 = d2.get("vel", None)

        if poses1 is None or poses2 is None:
            print(f"Skip: {filename} (poses missing)")
            continue

        poses_avg = weighted_arrays(poses1, poses2, best_weight)
        vel_avg = None
        if vel1 is not None and vel2 is not None:
            vel_avg = weighted_arrays(vel1, vel2, best_weight)

        poses_gt = d1.get("poses_gt", d1.get("Pgt", None))
        if poses_gt is None:
            poses_gt = d2.get("poses_gt", None)
        vel_gt = d1.get("vel_gt", None)
        if vel_gt is None:
            vel_gt = d2.get("vel_gt", None)

        out_path = os.path.join(args.dir, prefix + args.out_suffix)
        save_dict = {
            "poses": poses_avg[np.newaxis, ...],
        }
        if poses_gt is not None:
            save_dict["poses_gt"] = poses_gt
        if vel_avg is not None:
            save_dict["vel"] = vel_avg[np.newaxis, ...]
        if vel_gt is not None:
            save_dict["vel_gt"] = vel_gt

        np.savez(out_path, **save_dict)
        if poses_gt is not None:
            ate = calculate_ate_eval(poses_avg, poses_gt)
            rte = calculate_rte_eval(poses_avg, poses_gt, args.rte_duration)
            print(
                f"Saved {out_path} | weight: {best_weight:.4f} | "
                f"ATE: {ate:.6f} | RTE: {rte:.6f}"
            )
        else:
            print(f"Saved {out_path} | weight: {best_weight:.4f}")


if __name__ == "__main__":
    main()
