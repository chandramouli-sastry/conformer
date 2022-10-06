import model 
import torch 

torch.backends.cudnn.benchmark = True
L = 320000
EX_PER_DEVICE = 32

batch_size = torch.cuda.device_count() * EX_PER_DEVICE

# Initialize
m = model.ConformerEncoderDecoder(model.ConformerConfig())
wav = torch.randn(size=(2, L))
pad = torch.zeros_like(wav)
_ = m(wav, pad)

# Move to cuda 
# m = torch.nn.parallel.DistributedDataParallel(m.cuda()).train()
m = torch.nn.DataParallel(m.cuda()).train()
opt = torch.optim.Adam(m.parameters())

def train_step():
    opt.zero_grad(set_to_none=True)
    inp = torch.randn(size=(batch_size, L))
    pad = torch.zeros_like(inp)
    logits, logit_paddings = m(inp, pad)
    targets = torch.randint(low=1, high=1024, size=(logits.shape[0],256))
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
    print(loss.sum())

# 10 step of training
for i in range(10):
    print("Memory allocated: ",torch.cuda.memory_allocated()/1024**3,"GB")
    train_step()
