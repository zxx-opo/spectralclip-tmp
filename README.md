# SpectralCLIP

Source code, results, training logs, and figures for the SpectralCLIP paper.

## Repository Layout
- src/                       SpectralCLIP source (8 Python modules)
- scripts/master.py          Experiment driver
- results/                   99 JSON result files (pretrain + finetune + zero-shot + few-shot + cross-sensor)
- logs/                      Training jsonl logs
- corpus/                    Cached sentence-transformer text embeddings
- checkpoints/               Pretrained model weights (~26MB)
- figs/                      Paper figures (PDFs)
- paper_artifacts/           Earlier FreqTTA experiments and baselines
  - freqtta_src/             FreqTTA + 3 baselines (HybridSN, SpectralFormer, S2Mamba)
  - freqtta_scripts/         run_all.sh + analyze.py
  - freqtta_logs/            ~1.3MB training logs
  - freqtta_results/         48 in-domain + cross-scene results
- release/*.tar.gz           Bundled download tarballs

## Baselines Included
- HybridSN              paper_artifacts/freqtta_src/baselines.py
- SpectralFormer        paper_artifacts/freqtta_src/baselines.py
- S2Mamba               paper_artifacts/freqtta_src/baselines.py
- Tent / EATA / CoTTA / SMTTA / PhiTTA   paper_artifacts/freqtta_src/tta.py

## Datasets
PaviaU, PaviaC, IndianPines, Salinas (public). Download URLs in scripts/ or use the original USGS / ehu.eus links.

## Reproduce
    cd src && python3 run_experiment.py pretrain --seed 42 --epochs 60
    python3 run_experiment.py finetune --dataset PaviaU --seed 42 --ckpt ../checkpoints/pretrain_s42.pt
    python3 run_experiment.py zero_shot --dataset PaviaU --seed 42 --ckpt ../checkpoints/pretrain_s42.pt

## Citation
TBA
