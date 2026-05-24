"""Master driver for SpectralCLIP experiments.

Subcommands:
  - pretrain          : contrastive pretrain on aggregated 4-dataset corpus
  - finetune          : closed-set supervised finetune from a pretrained ckpt
  - zero_shot         : zero-shot evaluation using only text prompts
  - few_shot          : K-shot evaluation with textual-visual prototype
  - cross_sensor_zs   : cross-sensor zero-shot from src -> tgt
"""
import os
import sys
import json
import argparse

sys.path.insert(0, "/root/spectralclip-tmp")
import torch

from src.config import CFG, CKPT_DIR, RESULTS_DIR, DATASETS
from src.captions import all_prompts
from src.text_encoder import encode_captions
from src.model import SpectralCLIP
from src.data import build_pretrain_loader, build_indomain_loaders
from src.trainer import (
    set_seed, pretrain_contrastive, finetune_supervised,
    evaluate_supervised,
)
from src.evaluate_zs import (
    zero_shot_eval, few_shot_eval, cross_sensor_zero_shot, save_result,
)


def build_model():
    return SpectralCLIP(
        d_model=CFG.d_model, d_state=CFG.d_state, n_layers=CFG.n_layers,
        d_emb=CFG.d_emb, d_text=CFG.d_text,
    )


def cmd_pretrain(args):
    set_seed(args.seed)
    print(f"[pretrain] Building corpus loader (per_class={args.per_class})...")
    ds, loader = build_pretrain_loader(per_class=args.per_class, seed=args.seed,
                                       batch_size=args.batch_size)
    print(f"[pretrain] {len(ds)} (spectrum,caption) pairs, "
          f"{len(ds.captions)} unique captions across "
          f"{len(set(k[0] for k in ds.cap_keys))} datasets.")
    print(f"[pretrain] Encoding captions with sentence-transformer...")
    text_emb = encode_captions(ds.captions, device="cpu")
    print(f"[pretrain] text_emb shape={tuple(text_emb.shape)}")
    model = build_model()
    tag = f"pretrain_s{args.seed}"
    ckpt = pretrain_contrastive(
        model, loader, text_emb,
        epochs=args.epochs, lr=CFG.lr, wd=CFG.wd,
        lambda_phys=CFG.lambda_phys, tag=tag, device=CFG.device,
    )
    out = {"tag": tag, "ckpt": ckpt, "n_pairs": len(ds),
           "n_captions": len(ds.captions), "seed": args.seed,
           "epochs": args.epochs}
    save_result(tag, out)
    print(json.dumps(out, indent=2))


def cmd_finetune(args):
    set_seed(args.seed)
    model = build_model()
    if args.pretrain_ckpt:
        sd = torch.load(args.pretrain_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[finetune] loaded {args.pretrain_ckpt}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    tr, va, te, n_bands, n_cls = build_indomain_loaders(args.dataset, args.seed)
    tag = f"finetune_{args.dataset}_s{args.seed}"
    head, ckpt, info = finetune_supervised(
        model, tr, va, n_classes=n_cls, epochs=args.epochs,
        lr=CFG.lr, wd=CFG.wd, tag=tag, device=CFG.device,
    )
    te_m = evaluate_supervised(model, head, te, CFG.device)
    out = dict(tag=tag, dataset=args.dataset, seed=args.seed,
               pretrain_ckpt=args.pretrain_ckpt, ckpt=ckpt,
               best_va_oa=info["best_va_oa"], epochs_run=info["epochs_run"],
               test_OA=te_m["OA"], test_AA=te_m["AA"], test_kappa=te_m["kappa"],
               per_class=te_m["per_class"])
    save_result(tag, out)
    print(json.dumps(out, indent=2))


def cmd_zero_shot(args):
    set_seed(args.seed)
    model = build_model().to(CFG.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[zero_shot] loaded {args.ckpt}: "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    m = zero_shot_eval(model, args.dataset, seed=args.seed, device=CFG.device)
    tag = f"zs_{args.dataset}_s{args.seed}"
    save_result(tag, m)
    print(json.dumps(m, indent=2))


def cmd_few_shot(args):
    set_seed(args.seed)
    model = build_model().to(CFG.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    m = few_shot_eval(model, args.dataset, k_per_class=args.k,
                      seed=args.seed, device=CFG.device)
    tag = f"fs_{args.dataset}_k{args.k}_s{args.seed}"
    save_result(tag, m)
    print(json.dumps(m, indent=2))


def cmd_cross_sensor_zs(args):
    set_seed(args.seed)
    model = build_model().to(CFG.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    # finetune ckpt stores {"spec":..., "head":...}; load only spec_enc weights
    if isinstance(sd, dict) and "spec" in sd and "head" in sd:
        model.spec_enc.load_state_dict(sd["spec"])
        print(f"[cross_sensor] loaded spec_enc from finetune ckpt {args.ckpt}")
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[cross_sensor] loaded {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    m = cross_sensor_zero_shot(model, args.src, args.tgt,
                                seed=args.seed, device=CFG.device)
    tag = f"xs_{args.src}_to_{args.tgt}_s{args.seed}"
    save_result(tag, m)
    print(json.dumps(m, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pretrain")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--epochs", type=int, default=CFG.pretrain_epochs)
    p1.add_argument("--per_class", type=int, default=200)
    p1.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p1.set_defaults(func=cmd_pretrain)

    p2 = sub.add_parser("finetune")
    p2.add_argument("--dataset", required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--epochs", type=int, default=CFG.finetune_epochs)
    p2.add_argument("--pretrain_ckpt", default=None)
    p2.set_defaults(func=cmd_finetune)

    p3 = sub.add_parser("zero_shot")
    p3.add_argument("--dataset", required=True)
    p3.add_argument("--ckpt", required=True)
    p3.add_argument("--seed", type=int, default=42)
    p3.set_defaults(func=cmd_zero_shot)

    p4 = sub.add_parser("few_shot")
    p4.add_argument("--dataset", required=True)
    p4.add_argument("--k", type=int, required=True)
    p4.add_argument("--ckpt", required=True)
    p4.add_argument("--seed", type=int, default=42)
    p4.set_defaults(func=cmd_few_shot)

    p5 = sub.add_parser("cross_sensor_zs")
    p5.add_argument("--src", required=True)
    p5.add_argument("--tgt", required=True)
    p5.add_argument("--ckpt", required=True)
    p5.add_argument("--seed", type=int, default=42)
    p5.set_defaults(func=cmd_cross_sensor_zs)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()