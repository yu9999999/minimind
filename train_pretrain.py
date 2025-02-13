import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')

# 定义日志记录函数（Logger）:Logger 用于在非分布式模式下或当前设备为主节点时输出日志信息。
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)
# 学习率预热（warmup）+ 余弦退火（Cosine Annealing）。余弦退火参考后面的解释。warmup是通过调整warmup_iters值线性增加学习率，直到达到初始学习率。
# def get_lr(it, all, learning_rate):
#     warmup_iters = 0  # 可以根据需要调整
#     lr_decay_iters = all
#     min_lr = learning_rate / 10
    
#     if it < warmup_iters:
#         return learning_rate * it / warmup_iters
#     elif it > lr_decay_iters:
#         return min_lr
#     else:
#         decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
#         assert 0 <= decay_ratio <= 1
#         coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
#         return min_lr + coeff * (learning_rate - min_lr)

# 定义学习率调度函数（get_lr），主要是实现前期和后期的学习率较小，中期学习率较大的目的，使得模型保证快速收敛及减小最优解附近震荡
# 余弦退火策略动态调整学习率。
# 当math.cos(math.pi * current_step / total_steps)=0.8时，lr达到最大值，此时current_step / total_steps=0.2048附近
# 因为0.6+0.5*cos(pi * current_step / total_steps)从1.1慢慢减小到0.1，当0.6+0.5*cos(pi * current_step / total_steps)=1时lr达到最大值，下面是lr更新详细流程
# 0：lr(0.6+0.5*1)
# 1：lr(0.6+0.5*1)(0.6+0.5*cos(pi * 1 / total_steps))
# 2：lr(0.6+0.5*1)(0.6+0.5*cos(pi * 1 / total_steps))(0.6+0.5*cos(pi * 2 / total_steps))
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 定义单个训练周期函数（train_epoch）:每个 epoch 的训练过程：加载数据，前向传播，计算损失，反向传播，梯度裁剪，参数更新等。
# 核心技术点：自动混合精度训练（AMP，torch.cuda.amp）动态学习率调整、分布式数据并行（DDP）、多设备支持。
def train_epoch(epoch, wandb):
    # 损失函数定义：使用 nn.CrossEntropyLoss 作为损失函数，reduction='none' 表示损失不会被自动平均或求和，这在后续处理中非常有用
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # 计时开始：记录训练开始的时间，用于计算每个 epoch 的平均耗时。
    start_time = time.time()
    # 数据加载：通过一个循环遍历 train_loader，该加载器提供训练数据 (X, Y, loss_mask)。X 是输入数据，Y 是标签，loss_mask 用于加权损失，可能用于处理不平衡数据或忽略某些数据点。
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 设备分配：将输入数据、标签和损失掩码移动到指定的设备（如 GPU）。
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # 学习率调整
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 模型前向传播：使用上下文管理器 with ctx
        with ctx:
            # 计算模型的输出 res
            res = model(X)
            # 根据输出和标签计算损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 损失通过损失掩码加权，并考虑到辅助损失 aux_loss
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps
        # 使用 scaler（自动混合精度训练AMP）对损失进行缩放，然后执行反向传播。
        scaler.scale(loss).backward()
        # 梯度裁剪
        if (step + 1) % args.accumulation_steps == 0:
            # 取消之前通过 scaler.scale(loss) 对损失值进行的缩放，原地修改并存储在 optimizer 所管理的参数中的梯度值
            scaler.unscale_(optimizer)
            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新参数
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        # 打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()

# 定义模型初始化函数（init_model）：加载 tokenizer 和模型，支持两种加载方式：从预训练权重或直接从 transformers 库加载。
# 核心技术点：自定义模型加载、权重处理、参数统计。
def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

# 定义分布式模式初始化函数（init_distributed_mode）：初始化分布式训练的进程组、设备等。
# 核心技术点：NCCL 后端的分布式模式（dist.init_process_group），设置设备（torch.cuda.set_device）。
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 核心技术点：数据加载与打乱、混合精度训练、分布式训练、模型保存机制
    # 1、设置训练的各类超参数（例如 epochs, batch_size, learning_rate 等）。
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 模型保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 2、初始化分布式训练模式（DDP）
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    # 3、初始化模型和tokenizer，加载训练数据集，创建DataLoader。
    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    # 4、创建优化器（Adam）和混合精度训练的scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    # 5、迭代进行训练（调用 train_epoch 进行每个 epoch 的训练）
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
