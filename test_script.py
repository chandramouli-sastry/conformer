import model 
import torch 
import torch.distributed as dist
import os 


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

def train_step():
    opt.zero_grad(set_to_none=True)
    inp = torch.randn(size=(batch_size, L))
    pad = torch.zeros_like(inp)
    logits, logit_paddings = m(inp, pad)
    targets = torch.randint(low=0, high=1024, size=(logits.shape[0],256))
    target_paddings = torch.zeros_like(targets)
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    input_lengths = torch.einsum('bh->b', 1 - logit_paddings).long()
    target_lengths = torch.einsum('bh->b', 1 - target_paddings).long()
    loss = torch.nn.CTCLoss()(
        logprobs.permute(1, 0, 2),
        targets.long(),
        input_lengths,
        target_lengths)
    loss.backward()
    opt.step()
    return loss.sum().item()


# 10 step of training
for i in range(10):
    if rank==0:
        print(f"Memory allocated: ",torch.cuda.memory_allocated()/1024**3,"GB")
    loss = train_step()
    if rank==0:
        print(f"{loss}")
