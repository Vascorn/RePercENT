"""
End-to-end CLIP fine-tuning (OpenAI clip) with standard symmetric contrastive loss.
"""
import os, sys
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip as clip
from fine_tuning.clip_finetune_helpers import CLIP_ft_dataset, get_clip_encoded_batch, get_clip_logit_scale_exp, cosine_with_warmup, clip_symmetric_contrastive_loss, eval_clip, eval_clip_tensor
from src.utils.helpers import set_seed
from src.models.pretrained_encoders.clip_embeddings import CLIPHelper

import pandas as pd
import argparse
import wandb
import numpy as np
import gc


script_dir = os.path.dirname(os.path.abspath(__file__))


def prepare_finetune_clip(model, end2end= False, verbose= True):
    # end-to-end: unfreeze everything
    if end2end:
        for p in model.parameters():
            p.requires_grad = True
    else:
        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze projection heads
        if hasattr(model, "text_projection") and model.text_projection is not None:
            model.text_projection.requires_grad = True

        if hasattr(model, "visual") and hasattr(model.visual, "proj") and model.visual.proj is not None:
            model.visual.proj.requires_grad = True

        # Optional
        if hasattr(model, "logit_scale"):
            model.logit_scale.requires_grad = True

    if verbose:
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        print("Trainable params:", trainable)
    return model

def finetune_clip(
    model,
    train_loader,
    end2end = True,  # whether to unfreeze the entire model (True) or just the final projection layers (False)
    epochs=10,
    base_lr=2e-6,
    weight_decay=0.2,
    warmup_ratio=0.1,
    grad_clip=1.0,
    use_amp=True,
    seed= 2,
    device="cuda"
):
    set_seed(seed)
    
    model = model.to(device)
    model.train()

    model = prepare_finetune_clip(model, end2end=end2end, verbose=True)

    # AdamW is typical for CLIP fine-tuning
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    global_step = 0
    best_val = None
    best_state = None

    t0 = time.time()
    print_every = max(1, len(train_loader) // 5)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for step, out in enumerate(train_loader, start=1):
            images = out["image"]
            tokens = out["phrase"]
            images = images.to(device)
            tokens = tokens.to(device)
            global_step += 1

            # Update LR
            lr = cosine_with_warmup(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                text_features, image_features = get_clip_encoded_batch(model, tokens, images)
                
                # CLIP stores logit_scale in log-space; standard uses exp()
                logit_scale_exp = model.logit_scale.exp().clamp(1e-3, 100.0)

                loss = clip_symmetric_contrastive_loss(image_features, text_features, logit_scale_exp)

            scaler.scale(loss).backward()

            # Grad clipping (helps stability)
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

            if (step % print_every) == 0:
                avg = running / step
                print(
                    f"[epoch {epoch}/{epochs} step {step}/{len(train_loader)}] "
                    f"loss={avg:.4f} lr={lr:.2e} logit_scale_exp={float(logit_scale_exp):.3f}"
                )

        train_loss = running / max(1, len(train_loader))
        elapsed = time.time() - t0
        print(f"Epoch {epoch} done. train_loss={train_loss:.4f} elapsed={elapsed/60:.1f} min")

    return model


def aggregate_results(all_results):
    """
    all_results = list of result dictionaries (one per seed)
    """
    overall_accs = []
    for r in all_results:
        overall_accs.append(r['total_correct'] / r['total'])

    
    overall_mean = np.mean(overall_accs)
    overall_std  = np.std(overall_accs)

    # --- Per-type accuracy ---
    fig_types = all_results[0]['fig_s_type']
    
    per_type = {ft: [] for ft in fig_types}
    
    for r in all_results:
        for ft, acc in zip(r['fig_s_type'], r['Accuracy']):
            per_type[ft].append(acc)

    per_type_stats = {
        ft: {
            "mean": np.mean(vals),
            "std":  np.std(vals)
        }
        for ft, vals in per_type.items()
    }

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "per_type": per_type_stats
    }


def log_results_table(run, name, stats):
    """
    run   = wandb run
    name  = "With Definitions" etc
    stats = output of aggregate_results()
    """

    table = wandb.Table(columns=[
        "Setting",
        "Figurative Type",
        "Mean Accuracy",
        "Std Accuracy"
    ])

    # --- Overall row ---
    table.add_data(
        name,
        "OVERALL",
        stats["overall_mean"],
        stats["overall_std"]
    )

    # --- Per-type rows ---
    for ft, vals in stats["per_type"].items():
        table.add_data(
            name,
            ft,
            vals["mean"],
            vals["std"]
        )

    run.log({f"{name}_Results": table})

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP end-to-end on your dataset")
    parser.add_argument("--end2end", type=bool, default=False, help="Whether to unfreeze the entire model (True) or just the final projection layers (False)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--base_lr", type=float, default=2e-4, help="Base learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision (AMP) for faster training on CUDA")
    parser.add_argument("--base_seed", type=int, default=2, help="Random seed for reproducibility")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds for multiple runs")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="CLIP model variant to use (e.g., 'ViT-B-32', 'RN50', etc.)")
    parser.add_argument("--val_zero_shot", type=bool, default=False, help="Whether to evaluate zero-shot CLIP before fine-tuning")
    args = parser.parse_args()
    # Load datasets
    datasets_path = os.path.join(script_dir, "..", "data", "irfl", "datasets")
    test_df = pd.read_csv(os.path.join(datasets_path, "IRFL_test_detect_dataset_2.csv"))
    train_df = pd.read_csv(os.path.join(datasets_path, "IRFL_train_dataset_2.csv"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess_func = clip.create_model_and_transforms(args.clip_model, pretrained='openai', device=device)
    tokenizer = clip.get_tokenizer(args.clip_model)

    # Evaluate zero-shot CLIP before fine-tuning
    test_dataset = CLIP_ft_dataset(test_df, preprocess_func, tokenizer=tokenizer, test_mode=True)
    test_dataset.__repr__()
    test_dataloader = DataLoader(test_dataset, batch_size= 32, shuffle=False)
    
    if args.val_zero_shot:
        clip_helper = CLIPHelper(device=device, clip_model_name=args.clip_model)
        zs_results_w_def, zs_results_no_def, zs_results_def_only = eval_clip_tensor(clip_helper, test_dataloader, device)
        metrics_w_def = aggregate_results([zs_results_w_def])
        metrics_no_def = aggregate_results([zs_results_no_def])
        metrics_def_only = aggregate_results([zs_results_def_only])

        print("\n=== Zero-Shot CLIP Evaluation ===")
        print(f"With Definitions: Overall Acc = {metrics_w_def['overall_mean']:.4f}, Per Type: {metrics_w_def['per_type']}")
        print(f"Without Definitions: Overall Acc = {metrics_no_def['overall_mean']:.4f}, Per Type: {metrics_no_def['per_type']}")
        print(f"Definitions Only: Overall Acc = {metrics_def_only['overall_mean']:.4f}, Per Type: {metrics_def_only['per_type']}")
        
        

        # Log results
        wandb.init(project="CLIP_FineTuning_IRFL", name=f"CLIP_ZeroShot_{args.clip_model}", settings=wandb.Settings(disable_git=True))
        log_results_table(wandb.run, "With Definitions", metrics_w_def)
        log_results_table(wandb.run, "Without Definitions", metrics_no_def)
        log_results_table(wandb.run, "Definitions Only", metrics_def_only)

        wandb.finish()
    

    all_results_no_def = []
    all_results_w_def = []
    all_results_def_only = []

    for temp_seed_idx in range(args.n_seeds):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n\n=== Starting fine-tuning run with seed {args.base_seed + temp_seed_idx} ===")
        model, _, preprocess_func = clip.create_model_and_transforms(args.clip_model, pretrained='openai', device=device)
        tokenizer = clip.get_tokenizer(args.clip_model)
        # Assume you have a DataFrame `train_df` with columns "uuid" (image filename) and "phrase" (text)
        train_dataset = CLIP_ft_dataset(train_df, preprocess_func, tokenizer=tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        finetuned_model = finetune_clip(
            model=model,
            train_loader=train_loader,
            end2end=args.end2end,
            epochs=args.epochs,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_clip=args.grad_clip,
            use_amp=args.use_amp,
            seed=args.base_seed + temp_seed_idx,  # for reproducibility across seeds
            device=device
        )
        
        clip_helper = CLIPHelper(device=device, clip_model_name=args.clip_model, model=finetuned_model.to(device), preprocess_func=preprocess_func)
        temp_results_w_def, temp_results_no_def, temp_results_def_only = eval_clip_tensor(clip_helper, test_dataloader, device)

        all_results_w_def.append(temp_results_w_def)
        all_results_no_def.append(temp_results_no_def)
        all_results_def_only.append(temp_results_def_only)

    # Aggregate metrics across seeds
    metrics_w_def = aggregate_results(all_results_w_def)
    metrics_no_def = aggregate_results(all_results_no_def)
    metrics_def_only = aggregate_results(all_results_def_only)

    # Log results
    finetuned_name = f"CLIP_FineTuning_{args.clip_model}_seeds_{args.n_seeds}"
    finetuned_name += "_end2end" if args.end2end else "_proj_only"
    wandb.init(project="CLIP_FineTuning_IRFL", name=finetuned_name, settings=wandb.Settings(disable_git=True))
    log_results_table(wandb.run, "With Definitions", metrics_w_def)
    log_results_table(wandb.run, "Without Definitions", metrics_no_def)
    log_results_table(wandb.run, "Definitions Only", metrics_def_only)

    wandb.finish()

if __name__ == "__main__":
    main()