"""Master experiment runner for FAM-SSM."""
import os, sys, json, argparse
sys.path.insert(0, "/root/freqtta-hsi")
import torch

from src.config import CFG, RESULTS_DIR
from src.data import build_in_domain_loaders, build_transfer_loaders
from src.trainer import train_source, evaluate, set_seed
from src.model import FAMSSMHSI
from src.baselines import build_baseline
from src.tta import adapt_and_eval


def build_model(name, n_bands, n_classes):
    if name == "famssm" or name == "freqtta":
        return FAMSSMHSI(
            n_bands=n_bands, n_classes=n_classes,
            d_model=CFG.d_model, d_state=CFG.d_state,
            d_phi=CFG.d_phi, n_layers=CFG.n_layers,
            n_radial_bins=CFG.n_radial_bins, dropout=CFG.dropout,
        )
    return build_baseline(name, n_bands, n_classes, patch_size=CFG.patch_size)


def model_supports_recon(name):
    return name in ("famssm", "freqtta", "s2mamba")


def _save(out, tag):
    p = os.path.join(RESULTS_DIR, tag + ".json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return p


def run_in_domain(model_name, dataset, seed, epochs):
    set_seed(seed)
    tr, va, te, n_bands, n_cls = build_in_domain_loaders(dataset, seed)
    model = build_model(model_name, n_bands, n_cls)
    use_rec = model_supports_recon(model_name)
    tag = f"indomain_{dataset}_{model_name}_s{seed}"
    info, ckpt = train_source(
        model, tr, va, device=CFG.device, epochs=epochs,
        lr=CFG.lr, wd=CFG.wd, use_recon=use_rec,
        mask_ratio=CFG.mask_ratio, rec_w=CFG.rec_weight,
        log_prefix=tag, patience=25,
    )
    te_m = evaluate(model, te, CFG.device)
    out = dict(tag=tag, dataset=dataset, model=model_name, seed=seed,
               epochs_run=info["epochs_run"], best_va_oa=info["best_va_oa"],
               test_OA=te_m["OA"], test_AA=te_m["AA"], test_kappa=te_m["kappa"],
               per_class=te_m["per_class"], ckpt=ckpt)
    _save(out, tag)
    return out


def run_cross_scene(model_name, pair_name, seed, epochs,
                    tta_methods=("source_only", "tent", "eata",
                                 "cotta_lite", "smtta", "phitta")):
    set_seed(seed)
    s_tr, s_va, t_te, n_bands, n_cls = build_transfer_loaders(pair_name, seed)
    model = build_model(model_name, n_bands, n_cls)
    use_rec = model_supports_recon(model_name)
    tag_base = f"cross_{pair_name}_{model_name}_s{seed}"
    info, ckpt = train_source(
        model, s_tr, s_va, device=CFG.device, epochs=epochs,
        lr=CFG.lr, wd=CFG.wd, use_recon=use_rec,
        mask_ratio=CFG.mask_ratio, rec_w=CFG.rec_weight,
        log_prefix=tag_base + "_src", patience=25,
    )
    out = dict(tag=tag_base, pair=pair_name, model=model_name, seed=seed,
               src_epochs_run=info["epochs_run"],
               src_best_va_oa=info["best_va_oa"],
               src_ckpt=ckpt, tta={})
    for method in tta_methods:
        # smtta / phitta need a reconstruction decoder
        if method in ("smtta", "phitta") and not use_rec:
            continue
        try:
            m = adapt_and_eval(
                model, t_te, CFG.device, method=method,
                steps=CFG.tta_steps, lr=CFG.tta_lr,
                alpha=CFG.tta_alpha, tau=CFG.tta_tau,
                mask_ratio=CFG.mask_ratio,
            )
            out["tta"][method] = m
        except Exception as e:
            out["tta"][method] = {"error": str(e)}
    _save(out, tag_base)
    return out


def run_ablation(seed, epochs):
    """Ablations A1, A4, A5, A6 on famssm + S1_PaviaC2U."""
    set_seed(seed)
    pair = "P2_PaviaC_split"
    s_tr, s_va, t_te, n_bands, n_cls = build_transfer_loaders(pair, seed)
    results = {}

    def _train_eval(model, tag, **kwargs):
        info, _ = train_source(
            model, s_tr, s_va, device=CFG.device, epochs=epochs,
            lr=CFG.lr, wd=CFG.wd,
            use_recon=kwargs.get("use_recon", True),
            mask_ratio=CFG.mask_ratio, rec_w=CFG.rec_weight,
            log_prefix=tag, patience=20,
        )
        tta_kwargs = dict(steps=CFG.tta_steps, lr=CFG.tta_lr,
                          alpha=kwargs.get("alpha", CFG.tta_alpha),
                          tau=CFG.tta_tau, mask_ratio=CFG.mask_ratio)
        m_after = adapt_and_eval(model, t_te, CFG.device,
                                 method=kwargs.get("method", "phitta"), **tta_kwargs)
        m_before = adapt_and_eval(model, t_te, CFG.device, method="source_only")
        return dict(before=m_before, after=m_after, info=info)

    # A1: full FAM-SSM with phitta
    m = build_model("famssm", n_bands, n_cls)
    results["A1_full"] = _train_eval(m, f"abl_A1_full_s{seed}")

    # A4: full-LN Tent on FAM-SSM (instead of phitta)
    m = build_model("famssm", n_bands, n_cls)
    results["A4_full_ln_tent"] = _train_eval(
        m, f"abl_A4_fullLN_s{seed}", use_recon=True, method="tent")

    # A5: smtta (entropy only on all LN, with FAM-SSM) -- no recon at TTA
    m = build_model("famssm", n_bands, n_cls)
    results["A5_no_rec"] = _train_eval(
        m, f"abl_A5_norec_s{seed}", use_recon=False, method="eata")

    # A6: phitta w/o entropy term (alpha=0)
    m = build_model("famssm", n_bands, n_cls)
    results["A6_no_ent"] = _train_eval(
        m, f"abl_A6_noent_s{seed}", use_recon=True, method="phitta", alpha=0.0)

    tag = f"ablation_{pair}_s{seed}"
    _save(results, tag)
    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("in_domain")
    p1.add_argument("--model", required=True)
    p1.add_argument("--dataset", required=True)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--epochs", type=int, default=CFG.epochs)

    p2 = sub.add_parser("cross_scene_tta")
    p2.add_argument("--model", required=True)
    p2.add_argument("--pair", required=True)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--epochs", type=int, default=CFG.epochs)

    p3 = sub.add_parser("ablation")
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--epochs", type=int, default=CFG.epochs)

    args = ap.parse_args()
    out = None
    if args.cmd == "in_domain":
        out = run_in_domain(args.model, args.dataset, args.seed, args.epochs)
    elif args.cmd == "cross_scene_tta":
        out = run_cross_scene(args.model, args.pair, args.seed, args.epochs)
    elif args.cmd == "ablation":
        out = run_ablation(args.seed, args.epochs)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()