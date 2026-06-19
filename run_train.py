"""Training script — flat style, no main() function."""
import sys, time, json, random
from pathlib import Path
from datetime import datetime

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.bc_model import BCModel, NUM_BUTTONS, NUM_AXES
from preprocess.memory_dataset import load_all_sessions, collect_in_memory, InMemoryDataset

BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
SEQ_LEN = 4
MAX_SAMPLES = 50000
STEP = 3
VAL_SPLIT = 0.1
NUM_WORKERS = 0
AMP_ENABLED = True
SAVE_EVERY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints"

t_total = time.time()
print(f"Seq len: {SEQ_LEN} | Device: {DEVICE} | Epochs: {EPOCHS}", flush=True)

# ── Data ────────────────────────────────────────
print("Loading sessions...", flush=True)
t0 = time.time()
buffers = load_all_sessions()
print(f"  Loaded in {time.time()-t0:.0f}s", flush=True)

print("Collecting sequences...", flush=True)
t0 = time.time()
all_seqs = collect_in_memory(buffers, seq_len=SEQ_LEN, step=STEP, max_samples=MAX_SAMPLES)
print(f"  {len(all_seqs)} seqs in {time.time()-t0:.0f}s", flush=True)

random.shuffle(all_seqs)
n_val = max(1, int(len(all_seqs) * VAL_SPLIT))
train_seqs = all_seqs[n_val:]
val_seqs = all_seqs[:n_val]
print(f"  Train: {len(train_seqs)} | Val: {len(val_seqs)}", flush=True)

print("Creating DataLoader...", flush=True)
t0 = time.time()
train_ds = InMemoryDataset(train_seqs, buffers, augment=True)
val_ds = InMemoryDataset(val_seqs, buffers, augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
print(f"  Done in {time.time()-t0:.0f}s", flush=True)

# ── Model ───────────────────────────────────────
print("Creating model...", flush=True)
model = BCModel(num_frames=SEQ_LEN).to(DEVICE)
print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)

# pos_weight
pos = torch.zeros(NUM_BUTTONS)
for _, _, btns, _ in all_seqs:
    pos += torch.tensor(btns)
neg = len(all_seqs) - pos
pos_weight = neg.clone()
pos_weight[pos > 0] = neg[pos > 0] / pos[pos > 0]
pos_weight[pos == 0] = 0.0
pos_weight = pos_weight.to(DEVICE)

btn_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
axis_criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler("cuda") if AMP_ENABLED else None

# ── Train ───────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
best_val_loss = float("inf")
log_lines = []

print("Starting training...", flush=True)

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    model.train()
    train_btn_loss = 0.0
    train_axis_loss = 0.0
    optimizer.zero_grad()

    for step, (img, btns, axes) in enumerate(train_loader):
        img, btns, axes = img.to(DEVICE), btns.to(DEVICE), axes.to(DEVICE)
        with autocast("cuda", enabled=AMP_ENABLED):
            btn_logits, axes_pred = model(img)
            l_btn = btn_criterion(btn_logits, btns)
            l_axis = axis_criterion(axes_pred, axes)
            loss = l_btn + l_axis

        if AMP_ENABLED:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        train_btn_loss += l_btn.item()
        train_axis_loss += l_axis.item()

        if step % 500 == 0 and step > 0:
            print(f"  step {step}/{len(train_loader)}", flush=True)

    train_btn_loss /= len(train_loader)
    train_axis_loss /= len(train_loader)

    # Val
    model.eval()
    val_btn_loss = 0.0
    val_axis_loss = 0.0
    with torch.no_grad():
        for img, btns, axes in val_loader:
            img, btns, axes = img.to(DEVICE), btns.to(DEVICE), axes.to(DEVICE)
            btn_logits, axes_pred = model(img)
            val_btn_loss += btn_criterion(btn_logits, btns).item()
            val_axis_loss += axis_criterion(axes_pred, axes).item()

    val_btn_loss /= len(val_loader)
    val_axis_loss /= len(val_loader)
    val_total = val_btn_loss + val_axis_loss

    t_epoch = time.time() - t_ep
    msg = (f"Epoch {epoch:3d}/{EPOCHS} | "
           f"btn={train_btn_loss:.4f}/{val_btn_loss:.4f} "
           f"axis={train_axis_loss:.4f}/{val_axis_loss:.4f} | "
           f"{t_epoch:.0f}s")
    print(msg, flush=True)
    log_lines.append(msg)

    scheduler.step()

    if val_total < best_val_loss:
        best_val_loss = val_total
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if AMP_ENABLED else None,
            "loss": val_total,
            "seq_len": SEQ_LEN,
        }, OUTPUT_DIR / "best_stacked.pt")
        print(f"  -> best saved (loss={best_val_loss:.4f})", flush=True)

    if epoch % SAVE_EVERY == 0:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if AMP_ENABLED else None,
            "loss": val_total, "seq_len": SEQ_LEN,
        }, OUTPUT_DIR / f"epoch_stacked_{epoch:03d}.pt")

# Final save
torch.save({
    "epoch": EPOCHS, "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict() if AMP_ENABLED else None,
    "loss": val_total, "seq_len": SEQ_LEN,
}, OUTPUT_DIR / "last_stacked.pt")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(OUTPUT_DIR / f"train_log_{ts}.json", "w", encoding="utf-8") as f:
    json.dump({"config": {"batch_size": BATCH_SIZE, "epochs": EPOCHS,
                          "seq_len": SEQ_LEN, "train_seqs": len(train_seqs)},
               "log": log_lines}, f, indent=2, ensure_ascii=False)

print(f"\nDone! Best loss: {best_val_loss:.4f} | Total: {time.time()-t_total:.0f}s", flush=True)
