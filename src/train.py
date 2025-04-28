# ---------- train.py  (refactored) -----------------
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from chess_dataset import ChessDataset
from model import ChessNet


# ---------------- utilities ------------------------
def make_dataloaders(batch_size, num_workers=8):
    ds = ChessDataset("data/chess_data.h5", fraction=1.0)
    train_len = int(0.8 * len(ds))
    val_len   = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    common = dict(num_workers=num_workers,
                  pin_memory=True, persistent_workers=True,
                  prefetch_factor=16, batch_size=batch_size)
    return (
        DataLoader(train_ds, shuffle=True , **common),
        DataLoader(val_ds  , shuffle=False, **common)
    )


def train_one_model(cfg, train_loader, val_loader, device):
    """Train a single model described by cfg = {'name': 'A', 'filters':128, 'blocks':20}"""
    print(f"\n=== Training variant {cfg['name']}  "
          f"({cfg['filters']} filters | {cfg['blocks']} residual blocks) ===")

    model = ChessNet(num_filters=cfg['filters'],
                     num_res_blocks=cfg['blocks'])
    model.to(device)
    model = torch.compile(model)

    crit_v = nn.MSELoss()
    crit_p = nn.CrossEntropyLoss()
    opt    = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    num_epochs = 10
    sched  = CosineAnnealingLR(opt, T_max=num_epochs)

    scaler = torch.amp.GradScaler()
    accum_steps = 2
    best_val = float('inf')

    out_dir = f"models/{cfg['name']}"
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(num_epochs):
        tic = time.time()
        model.train();  train_loss = 0.0
        opt.zero_grad()

        for b, (x, tgt) in enumerate(train_loader):
            x, tgt = x.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            pol_t, val_t = tgt[:,0].long(), tgt[:,1]

            with torch.amp.autocast(device_type='cuda'):
                pol_p, val_p = model(x)
                loss = (crit_p(pol_p, pol_t) + crit_v(val_p.squeeze(), val_t)) / accum_steps

            scaler.scale(loss).backward()

            if (b+1) % accum_steps == 0 or (b+1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()

            train_loss += loss.item()*accum_steps

        # ---- validation ----
        model.eval();   val_loss = 0.0; correct = tot = 0
        with torch.no_grad():
            for x, tgt in val_loader:
                x, tgt = x.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
                pol_t, val_t = tgt[:,0].long(), tgt[:,1]

                with torch.amp.autocast(device_type='cuda'):
                    pol_p, val_p = model(x)
                    loss = crit_p(pol_p, pol_t) + crit_v(val_p.squeeze(), val_t)
                val_loss += loss.item()

                pred = pol_p.argmax(1)
                tot += pol_t.size(0);   correct += (pred == pol_t).sum().item()

        sched.step()
        toc = time.time()

        avg_tr = train_loss/len(train_loader)
        avg_va = val_loss /len(val_loader)
        acc    = 100*correct/tot
        print(f"Ep {epoch:2d}: "
              f"train {avg_tr:6.4f} | val {avg_va:6.4f} | acc {acc:5.2f}% "
              f"| {toc-tic:5.1f}s")

        # keep the best model
        if avg_va < best_val:
            best_val = avg_va
            torch.save(model.state_dict(), f"{out_dir}/best.pth")

    # final save
    torch.save(model.state_dict(), f"{out_dir}/last.pth")
    print(f"✔ finished variant {cfg['name']}  (best val loss {best_val:.4f})")


# ---------------- experiment runner ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="variants.json")
    parser.add_argument("--batch",  type=int, default=4096)
    args = parser.parse_args()

    with open(args.config) as f:
        variants = json.load(f)   # e.g. [{"name":"A","filters":128,"blocks":10}, ... ]

    train_loader, val_loader = make_dataloaders(args.batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True 

    for cfg in variants:
        train_one_model(cfg, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
# ---------------------------------------------------
