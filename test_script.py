import model 
import torch 
import torch.distributed as dist
import os 
from Libri_dataset import LibriDataset


torch.backends.cudnn.benchmark = False
use_pytorch_ddp = 'LOCAL_RANK' in os.environ
rank = int(os.environ['LOCAL_RANK']) if use_pytorch_ddp else 0
device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
torch.cuda.set_device(rank)
if use_pytorch_ddp:
    dist.init_process_group('nccl')
L = 320000
EX_PER_DEVICE = 32

# Initialize
m = model.ConformerEncoderDecoder(model.ConformerConfig())
wav = torch.randn(size=(2, L))
pad = torch.zeros_like(wav)
_ = m(wav, pad)

# Move to cuda 
if use_pytorch_ddp:
    m = torch.nn.parallel.DistributedDataParallel(m.cuda(rank),device_ids=[rank], output_device=rank).train()
    batch_size = EX_PER_DEVICE
else:
    m = torch.nn.DataParallel(m.cuda()).train()
    batch_size = EX_PER_DEVICE * n_gpus
opt = torch.optim.Adam(m.parameters())

# Create Dataset
ds = LibriDataset(split="train-clean-100", data_dir="/mnt/disks/librispeech_processed/work_dir/data")
sampler = None
USE_PYTORCH_DDP = use_pytorch_ddp
train = True
N_GPUS = n_gpus 
RANK = rank 

if USE_PYTORCH_DDP:
    if train:
        sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=N_GPUS, rank=RANK, shuffle=True, seed=0)
    else:
        sampler = data_utils.DistributedEvalSampler(
        ds, num_replicas=N_GPUS, rank=RANK, shuffle=False)
dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=not USE_PYTORCH_DDP and train,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=train)

def train_step(inp, pad, targets, target_paddings):
    opt.zero_grad(set_to_none=True)
    # inp = torch.randn(size=(batch_size, L))
    # pad = torch.zeros_like(pad)
    # print(inp.device)
    logits, logit_paddings = m(inp.to(device), pad.to(device))
    if rank==0:
        print(f"Memory allocated after fwd: ",torch.cuda.memory_allocated()/1024**3,"GB")
    # logit_paddings = torch.zeros_like(logit_paddings)
    # targets = torch.randint(low=0, high=1024, size=(logits.shape[0],256))
    # target_paddings = torch.zeros_like(targets)
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    input_lengths = torch.einsum('bh->b', 1 - logit_paddings).long()
    target_lengths = torch.einsum('bh->b', 1 - target_paddings).long()
    loss = torch.nn.CTCLoss(reduction='none').to(device)(
        logprobs.permute(1, 0, 2),
        targets.long(),
        input_lengths,
        target_lengths)
    loss = loss.sum()/target_lengths.sum()
    loss.backward()
    opt.step()
    return loss.sum().item()


# 10 step of training
for i, batch in enumerate(dataloader):
    if i==100:
        break
    if rank==0:
        print(f"Memory allocated: ",torch.cuda.memory_allocated()/1024**3,"GB")
    loss = train_step(batch[0][0],batch[0][1],batch[1][0],batch[1][1])
    if rank==0:
        print(f"{loss}")
    torch.cuda.empty_cache()
    # exit()
