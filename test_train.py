import sys; sys.path.insert(0, '.')
import time, random, torch
from torch.utils.data import DataLoader
from models.bc_model import BCModel
from preprocess.memory_dataset import load_all_sessions, collect_in_memory, InMemoryDataset

print('Loading...', flush=True); t0=time.time()
buffers = load_all_sessions()
seqs = collect_in_memory(buffers, seq_len=4, step=3, max_samples=50000)
random.shuffle(seqs)
n_val = max(1, int(len(seqs)*0.1))
train_seqs = seqs[n_val:]
print(f'{len(train_seqs)} train seqs in {time.time()-t0:.1f}s', flush=True)

print('Creating DataLoader...', flush=True); t0=time.time()
ds = InMemoryDataset(train_seqs, buffers, augment=True)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
print(f'Done in {time.time()-t0:.1f}s', flush=True)

print('Model...', flush=True)
model = BCModel(num_frames=4).to('cuda')
print(f'Model ready, starting loop...', flush=True)

for step, (img, btns, axes) in enumerate(dl):
    print(f'step {step}: img={img.shape}', flush=True)
    if step >= 2:
        break

print('ALL OK', flush=True)
