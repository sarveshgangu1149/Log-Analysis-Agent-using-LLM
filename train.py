import os
import csv
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from model import LogLLM
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler
from torch import optim


n_epochs_1 = int(os.environ.get('N_EPOCHS_1', 2))
n_epochs_2_1 = int(os.environ.get('N_EPOCHS_2_1', 2))
n_epochs_2_2 = int(os.environ.get('N_EPOCHS_2_2', 2))
n_epochs_3 = int(os.environ.get('N_EPOCHS_3', 4))
dataset_name = 'LOCAL'
batch_size = int(os.environ.get('BATCH_SIZE', 16))
# Slightly larger micro-batch on MPS/CPU; override via env if needed
micro_batch_size = int(os.environ.get('MICRO_BATCH_SIZE', 2))
gradient_accumulation_steps = batch_size // micro_batch_size


lr_1 = 5e-4
lr_2_1 = 5e-4
lr_2_2 = 5e-5
lr_3 = 5e-5
# Increase token budgets; override via env if needed
max_content_len = int(os.environ.get('MAX_CONTENT_LEN', 96))
max_seq_len = int(os.environ.get('MAX_SEQ_LEN', 96))

# Use local CSV prepared from dataset/input_data.json
data_path = str(Path(__file__).parent / 'dataset' / 'train.csv')

min_less_portion = 0.3

# Use local model folders; prefer TinyLlama if present, allow override via LLM_PATH
ROOT_DIR = Path(__file__).parent
Bert_path = str(ROOT_DIR / 'bert-base-uncased')
_llm_env = os.environ.get('LLM_PATH')
if _llm_env:
    Llama_path = _llm_env
else:
    tiny_path = ROOT_DIR / 'TinyLlama-1.1B-Chat-v1.0'
    llama3_path = ROOT_DIR / 'Meta-Llama-3-8B'
    if tiny_path.exists():
        Llama_path = str(tiny_path)
    elif llama3_path.exists():
        Llama_path = str(llama3_path)
    else:
        # fallback to TinyLlama name (user may download later)
        Llama_path = str(tiny_path)

ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

# Auto-detect device; prefer CUDA, then Apple MPS, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'n_epochs_1: {n_epochs_1}\n'
f'n_epochs_2_1: {n_epochs_2_1}\n'
f'n_epochs_2_2: {n_epochs_2_2}\n'
f'n_epochs_3: {n_epochs_3}\n'
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'micro_batch_size: {micro_batch_size}\n'
f'lr_1: {lr_1}\n'
f'lr_2_1: {lr_2_1}\n'
f'lr_2_2: {lr_2_2}\n'
f'lr_3: {lr_3}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'min_less_portion: {min_less_portion}\n'
f'device: {device}')

def print_number_of_trainable_model_parameters(model):
    params = set()
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            params.add(param)
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return params



def trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr, phase_name: str):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    normal_tokens = model.Llama_tokenizer('The sequence is normal.')['input_ids']
    anomalous_tokens = model.Llama_tokenizer('The sequence is anomalous.')['input_ids']
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f'scheduler_step: {scheduler_step}')

    steps = 0
    # Prepare simple CSV logger
    runs_dir = Path(__file__).parent / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / 'train_metrics.csv'
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['phase', 'epoch', 'loss', 'acc', 'last_lr', 'steps', 'samples'])
    for epoch in range(int(n_epochs)):
        total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0
        epoch_start = time.time()

        pbar = tqdm(dataloader, desc='Epoch {}/{}'.format(epoch, n_epochs))
        for i_th, bathc_i in enumerate(pbar):
            steps += 1

            inputs= bathc_i['inputs']
            seq_positions= bathc_i['seq_positions']
            labels = bathc_i['labels']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs, targets = model.train_helper(inputs, seq_positions, labels)
            # MPS/CPU often require float32 for loss; ensure dtypes are correct
            outputs = outputs.float()
            targets = targets.long()
            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

            loss.backward()
            # print(loss)

            if ((i_th + 1) % gradient_accumulation_steps == 0) or ((i_th + 1) == len(dataloader)):
                # optimizer the net
                optimizer.step()  # 更新网络参数
                optimizer.zero_grad()  # reset grdient # 清空过往梯度

            acc_mask = torch.zeros_like(targets,device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
            total_acc_count += acc_mask.sum()

            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)

            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()
            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss = loss.item() * gradient_accumulation_steps)

            if steps % 10000 ==0:   # every 10000 steps, print loss and acc
                train_loss_epoch = train_loss / total_count
                train_acc_epoch = total_acc / total_acc_count
                print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                      f"[loss: {train_loss_epoch:3f}]"
                      f"[acc: {train_acc_epoch:3f}]")

                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            train_acc_epoch = total_acc / total_acc_count
            print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]"
                  f"[acc: {train_acc_epoch:3f}]")
            # Append epoch metrics
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    phase_name,
                    epoch + 1,
                    f"{train_loss_epoch:.6f}",
                    f"{train_acc_epoch:.6f}",
                    f"{scheduler.get_last_lr()[0]:.8f}",
                    steps,
                    total_count,
                ])

if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path, drop_duplicates=False)

    model = LogLLM(Bert_path, Llama_path, device = device, max_content_len = max_content_len, max_seq_len = max_seq_len)
    # model = LogLLM(Bert_path, Llama_path, ft_path= ft_path, device = device, max_content_len = max_content_len, max_seq_len = max_seq_len)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)

    # For small datasets, cap sampler size to dataset size to avoid errors
    max_samples_phase1 = min(1000, len(dataset))
    num_workers = int(os.environ.get('NUM_WORKERS', 2))

    dataloader_max_samples = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion, max_samples=max_samples_phase1),
        collate_fn=collator,
        drop_last=True
    )
    # phase 1
    print("*" * 10 + "Start training Llama" + "*" * 10)
    model.set_train_only_Llama()
    trainModel(model, dataloader_max_samples, gradient_accumulation_steps, n_epochs_1, lr_1, phase_name='phase1_llama')
    del dataloader_max_samples

    # Limit samples for small custom datasets to avoid huge epoch sizes
    total_samples = len(dataset)
    max_samples_phase2 = min(5000, total_samples)

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion, max_samples=max_samples_phase2),
        collate_fn=collator,
        drop_last=True
    )
    # phase 2-1
    print("*" * 10 + "Start training projector" + "*" * 10)
    model.set_train_only_projector()
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_1, lr_2_1, phase_name='phase2_projector')
    # phase 2-2
    print("*" * 10 + "Start training projector and Bert" + "*" * 10)
    model.set_train_projectorAndBert()
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_2, lr_2_2, phase_name='phase2_proj_bert')
    # phase 3
    model.set_finetuning_all()
    print("*" * 10 + "Start training entire model" + "*" * 10)
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_3, lr_3, phase_name='phase3_full')

    model.save_ft_model(ft_path)
