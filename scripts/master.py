import os, sys, json, subprocess
sys.path.insert(0, "/root/spectralclip-tmp")
RES = "/root/spectralclip-tmp/results"
CKPT = "/root/spectralclip-tmp/checkpoints"
SEEDS = [42, 43, 44]
PRETRAIN_EPOCHS = 60
FINETUNE_EPOCHS = 40
DATASETS = ["PaviaU", "PaviaC", "IndianPines", "Salinas"]

def run(tag, cmd):
    fp = os.path.join(RES, tag + ".json")
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        print(f"[SKIP] {tag}")
        return
    print(f"[RUN ] {tag}")
    r = subprocess.call(["python3"] + cmd, cwd="/root/spectralclip-tmp")
    print(f"[{tag}] exit={r}")

def main():
    for s in SEEDS:
        run(f"pretrain_s{s}", ["src/run_experiment.py", "pretrain", "--seed", str(s), "--epochs", str(PRETRAIN_EPOCHS), "--per_class", "200"])
    for s in SEEDS:
        for d in DATASETS:
            run(f"finetune_{d}_s{s}", ["src/run_experiment.py", "finetune", "--dataset", d, "--seed", str(s), "--epochs", str(FINETUNE_EPOCHS), "--pretrain_ckpt", f"{CKPT}/pretrain_s{s}.pt"])
    for s in SEEDS:
        for d in DATASETS:
            run(f"zs_{d}_s{s}", ["src/run_experiment.py", "zero_shot", "--dataset", d, "--seed", str(s), "--ckpt", f"{CKPT}/pretrain_s{s}.pt"])
    for s in SEEDS:
        for d in DATASETS:
            for k in [1, 5, 10]:
                run(f"fs_{d}_k{k}_s{s}", ["src/run_experiment.py", "few_shot", "--dataset", d, "--k", str(k), "--seed", str(s), "--ckpt", f"{CKPT}/pretrain_s{s}.pt"])
    for s in SEEDS:
        for src in DATASETS:
            for tgt in DATASETS:
                if src == tgt: continue
                run(f"xs_{src}_to_{tgt}_s{s}", ["src/run_experiment.py", "cross_sensor_zs", "--src", src, "--tgt", tgt, "--seed", str(s), "--ckpt", f"{CKPT}/finetune_{src}_s{s}.pt"])
    print("ALL DONE")

if __name__ == "__main__":
    main()
