"""Aggregate JSON results into Markdown tables for FAM-SSM."""
import os, json, glob, statistics as st
from collections import defaultdict

RES = "/root/freqtta-hsi/results"

def _ms(xs):
    if not xs:                return float("nan"), float("nan")
    if len(xs) == 1:          return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def aggregate_in_domain():
    rows = defaultdict(lambda: defaultdict(list))
    for fp in sorted(glob.glob(f"{RES}/indomain_*.json")):
        d = json.load(open(fp))
        rows[d["dataset"]][d["model"]].append(d["test_OA"])
    print("## In-domain OA (mean +/- std)\n")
    print("| dataset | famssm | s2mamba | spectralformer | hybridsn |")
    print("|---------|--------|---------|----------------|----------|")
    for ds in ["PaviaU", "PaviaC", "IndianPines", "Salinas"]:
        cells = [ds]
        for m in ["famssm", "s2mamba", "spectralformer", "hybridsn"]:
            mu, sd = _ms(rows[ds].get(m, []))
            cells.append(f"{mu*100:5.2f}+/-{sd*100:.2f}")
        print("| " + " | ".join(cells) + " |")
    print()


def aggregate_cross_scene():
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fp in sorted(glob.glob(f"{RES}/cross_*.json")):
        d = json.load(open(fp))
        for meth, m in d.get("tta", {}).items():
            if isinstance(m, dict) and "OA" in m:
                rows[d["pair"]][d["model"]][meth].append(m["OA"])
    methods = ["source_only", "tent", "eata", "cotta_lite", "smtta", "phitta"]
    for pair, model_d in rows.items():
        print(f"## Cross-scene target OA on {pair}\n")
        print("| backbone | " + " | ".join(methods) + " |")
        print("|" + "|".join(["---"] * (len(methods) + 1)) + "|")
        for m in ["famssm", "s2mamba", "spectralformer", "hybridsn"]:
            row = [m]
            for meth in methods:
                xs = model_d.get(m, {}).get(meth, [])
                if not xs:
                    row.append("-")
                else:
                    mu, sd = _ms(xs)
                    row.append(f"{mu*100:5.2f}+/-{sd*100:.2f}")
            print("| " + " | ".join(row) + " |")
        print()


def aggregate_ablation():
    keys = ["A1_full", "A4_full_ln_tent", "A5_no_rec", "A6_no_ent"]
    rows = defaultdict(list)
    for fp in sorted(glob.glob(f"{RES}/ablation_*.json")):
        d = json.load(open(fp))
        for k in keys:
            if k in d:
                rows[k].append((d[k]["before"]["OA"], d[k]["after"]["OA"]))
    print("## Ablation on S1_PaviaC2U (target OA before / after)\n")
    print("| variant | before | after | delta |")
    print("|---------|--------|-------|-------|")
    for k in keys:
        if not rows[k]:
            print(f"| {k} | - | - | - |"); continue
        bs, as_ = zip(*rows[k])
        bmu, bsd = _ms(list(bs)); amu, asd = _ms(list(as_))
        d = amu - bmu
        print(f"| {k} | {bmu*100:5.2f}+/-{bsd*100:.2f} | "
              f"{amu*100:5.2f}+/-{asd*100:.2f} | {d*100:+.2f} |")
    print()


if __name__ == "__main__":
    aggregate_in_domain()
    aggregate_cross_scene()
    aggregate_ablation()