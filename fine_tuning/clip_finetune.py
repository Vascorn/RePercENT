"""
End-to-end CLIP fine-tuning (OpenAI clip) with standard symmetric contrastive loss.

Assumptions:
- You have a PyTorch Dataset that returns (PIL.Image, text_str).
- You already have: model, preprocess_func = clip.load('ViT-B/32', device=device)

Notes for small data (~2k):
- Keep LR small (e.g., 1e-6 to 5e-6) for true end-to-end.
- Use AMP + grad clipping.
- Validate often with your existing compute_accuracy() (optional hook below).
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
from fine_tuning.clip_finetune_helpers import CLIP_ft_dataset, get_clip_encoded_batch, get_clip_logit_scale_exp, cosine_with_warmup, clip_symmetric_contrastive_loss
from src.utils.helpers import set_seed


def finetune_clip_end2end(
    model,
    train_loader,         # optional callable: () -> dict or float
    epochs=10,
    base_lr=2e-6,
    weight_decay=0.2,
    warmup_ratio=0.1,
    grad_clip=1.0,
    use_amp=True
):
    set_seed(seed)
    
    model = model.to(device)
    model.train()

    # TRUE end-to-end: unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    # AdamW is typical for CLIP fine-tuning
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    total_steps = epochs * len(loader)
    warmup_steps = int(total_steps * warmup_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    global_step = 0
    best_val = None
    best_state = None

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for step, (images, tokens) in enumerate(train_loader, start=1):
            global_step += 1

            # Update LR
            lr = cosine_with_warmup(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                image_features, text_features = get_clip_encoded_batch(model, tokens, images)

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
                    f"[epoch {epoch}/{epochs} step {step}/{len(loader)}] "
                    f"loss={avg:.4f} lr={lr:.2e} logit_scale_exp={float(logit_scale_exp):.3f}"
                )

        train_loss = running / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"Epoch {epoch} done. train_loss={train_loss:.4f} elapsed={elapsed/60:.1f} min")

    return model



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess_func = clip.load('ViT-B/32', device=device)

    # Assume you have a DataFrame `train_df` with columns "uuid" (image filename) and "phrase" (text)
    train_dataset = CLIP_ft_dataset(train_df, preprocess_func)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    finetuned_model = finetune_clip_end2end(
        model=model,
        train_loader=train_loader,
        epochs=10,
        base_lr=2e-6,
        weight_decay=0.2,
        warmup_ratio=0.1,
        grad_clip=1.0,
        use_amp=True
    )

if __name__ == "__main__":
    main()