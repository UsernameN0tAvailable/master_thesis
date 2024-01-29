import os
import argparse
from typing import Optional, Dict, List, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, f1_score
import torch
from torch import nn, Tensor
from torch.types import Number
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.dino import DinoFeatureClassifier
from models.mlp_header import MLPHeader
from models.streaming.top import TopCNN
from models.streaming.bottom import BottomCNN, Vit, ResNet
from models.streaming.scnn import StreamingNet
import csv
import numpy as np
import math
import ssl
import re

# distribute
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import wandb
from pipeline_utils import get_dataloaders, Logger, Optimizer
from torch.nn import SyncBatchNorm

# fix ssl pytorch bug
ssl._create_default_https_context = ssl._create_unverified_context

def step(model, optimizer: Optional[Optimizer], criterion, dataloader, device: str, rank: int, device_count: int, threshold: float = 0.5, average: Optional[str] ="weighted"):

    running_loss: Tensor = torch.tensor(0.0, device=device)
    true: List[np.ndarray] = []
    preds: List[np.ndarray] = []

    for images, labels, clinical_data in dataloader:
        labels: Tensor = labels.to(device)
        images: Tensor = images.to(device)
        clinical_data: Tensor = clinical_data.to(device)

        if optimizer is not None: optimizer.optimizer.zero_grad()

        output, loss = model.module.step(images, labels, criterion, optimizer, clinical_data)
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(output.data[:, 1]) 
        predicted = (probs > threshold).long()
        
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

        if optimizer is not None:
            optimizer.optimizer.step()
            optimizer.scheduler.step()
            optimizer.optimizer.zero_grad() 

    dist.all_reduce(running_loss, op=dist.ReduceOp.AVG)
    epoch_loss: float = running_loss.item() / len(dataloader.dataset)

    true_tensor: Tensor = torch.tensor(true, device=device)
    preds_tensor: Tensor = torch.tensor(preds, device=device)

    gathered_true_tensors: List[Tensor] = [torch.zeros_like(true_tensor) for _ in range(device_count)]
    gathered_preds_tensors: List[Tensor] = [torch.zeros_like(preds_tensor) for _ in range(device_count)]

    dist.all_gather(gathered_true_tensors, true_tensor)
    dist.all_gather(gathered_preds_tensors, preds_tensor)

    if rank == Logger.logging_rank:
        gathered_trues = torch.cat(gathered_true_tensors, dim=0).cpu().numpy()
        gathered_preds = torch.cat(gathered_preds_tensors, dim=0).cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(gathered_trues, gathered_preds, average=average, zero_division=0.0)
        avg_f1 = f1_score(gathered_trues, gathered_preds, average="weighted", zero_division=0.0)
        return epoch_loss, precision, recall, f1, avg_f1

    return None, None, None, None, None


# validation
def check_batch_size(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def validate_model_and_extract(s):
    # Check for vit[*] format
    vit_match = re.match(r'^(vit)\[(\d+)\]$', s)
    if vit_match:
        number = int(vit_match.group(2))
        if number > 0 and number % 16 == 0:
            return {'type': 'vit', 'value': number}
        else:
            raise ValueError(f"Invalid value for 'vit': {number}. It must be a positive integer divisible by 16.")

    # Check for stream[m, n] format
    stream_match = re.match(r'^stream\[(vit|cnn|resnet)->(vit|cnn|resnet|vit\[U\]|resnet\[U\]), (\d+)\]$', s)
    if stream_match:
        top_m = stream_match.group(1)
        bottom_m = stream_match.group(2)
        n = int(stream_match.group(3))

        if n <= 0:
            raise ValueError(f"Invalid value for 'n': {n}. It must be a positive integer.")
        
        if n % 16 != 0:
            raise ValueError(f"Invalid combination: stream[{top_m}|{bottom_m}, {n}]. When 'm' is 'vit', 'n' must be divisible by 16.")
        
        return {'type': 'stream', 'top': top_m, 'bottom': bottom_m, 'value': n}

    header_match = re.match(r'^header_only', s)
    if header_match:
        return {'type': 'header_only'}

    raise ValueError("Model type not available\nPossible types:\n- vit[<tile_size>]\n- stream[<cnn | vit | resnet | unet>|<cnn | vit | resnet | unet>, <patch_size>]\n- header_only")


def create_model(param: Dict[str, Any], device: str, has_clinical_data: bool = False) -> DinoFeatureClassifier | StreamingNet | MLPHeader:

    model: Optional[DinoFeatureClassifier | StreamingNet] = None
    Logger.log(f"Model: {param}")

    if param['type'] == 'vit':
        return DinoFeatureClassifier(clinical_data=has_clinical_data)
    elif param['type'] == 'stream':

        top_param = param['top']
        bottom_param = param['bottom']

        if top_param == 'cnn':
            top_net = TopCNN(has_clinical_data)
        else:
            raise ValueError(f'No {top_param} top net available!!')

        if bottom_param == 'cnn':
            bottom_net = BottomCNN()
        elif bottom_param == 'vit':
            bottom_net = Vit(32)
        elif bottom_param == 'vit[U]':
            bottom_net = Vit(32, False)
        elif bottom_param == 'resnet':
            bottom_net = ResNet(32)
        elif bottom_param == 'resnet[U]':
            bottom_net = ResNet(32, False)
        else:
            raise ValueError(f'No {bottom_param} bottom net available!!')

        return StreamingNet(top_net.to(device), bottom_net.to(device), param['value']) 
    elif param['type'] == 'header_only':
        return MLPHeader()
    else:
        raise ValueError(f'{param} Not Implemented')

    return model


def load_clinical_data(path: str) -> Dict[str, np.ndarray]:

    out: Dict[str, np.ndarray] = {}

    with open(path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            [name, arg0, arg1, arg2, arg3] = row
            out[name.replace(".", "dot")] = np.array([
                float(1.0 if arg0 == "Yes" or arg0 == "yes" else 0.0),
                float(arg1),
                float(arg2),
                float(arg3)
                ], dtype=np.float32)
    return out


def main():

    print("0")

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--type", type=str, default="vit[2048]")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--batch_size", type=check_batch_size, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--oversample", type=float, default=0.5)
    parser.add_argument("-t", action="store_true", default=False, help="Test only run")
    parser.add_argument("-i", action="store_true", default=False, help="Train with smaller images")
    parser.add_argument("-p", action="store_true", default=False, help="Pin Memory for more efficient memory management")
    parser.add_argument("--clinical_data", type=str, default=None, help="Additional clinical data file path for the classification net")
    parser.add_argument("--threshold", type=float, default=0.5, help="last layer's decision threshold")

    print("1")

    args = parser.parse_args()

    threshold: float = args.threshold 

    clinical_data: Optional[Dict[str, np.ndarray]] = None;

    if args.clinical_data is not None:
        clinical_data = load_clinical_data(args.clinical_data)

    if args.t and args.i:
        raise ValueError("Cannot use reduced image size for testing!!")

    model_type = validate_model_and_extract(args.type)

    if model_type['type'] == 'header_only' and clinical_data is None:
        raise ValueError("Cannot Execture with header only without clinical data")

    print("1.1")

    dist.init_process_group(backend='nccl', init_method='env://')

    print("1.2")

    rank: int = int(os.environ["LOCAL_RANK"])

    print("1.3")

    device_count: int = torch.cuda.device_count()
    device: str = str(torch.device(f'cuda:{rank}') if args.device == 'cuda' else torch.device('cpu'))

    print("1.4")

    # Memory Load Management
    local_gpu_memory: int = int(torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2))

    print("1.5")

    all_local_gpu_memories: Tensor = torch.zeros(device_count, dtype=torch.int64, device=device) 
    all_local_gpu_memories[rank] = torch.tensor(local_gpu_memory, dtype=torch.int64, device=device)

    print("1.6")

    dist.all_reduce(all_local_gpu_memories, op=dist.ReduceOp.SUM)
    tot_gpu_memory: Number = all_local_gpu_memories.sum().item()

    print("1.7")

    local_batch_size: int = int(args.batch_size * (local_gpu_memory / tot_gpu_memory))

    print("1.8")

    tot_batch_size_tensor: Tensor = torch.tensor(local_batch_size, dtype=torch.int, device=device)
    dist.all_reduce(tot_batch_size_tensor, op=dist.ReduceOp.SUM)
    tot_batch_size: Number = tot_batch_size_tensor.item()

    # Select GPU with lowest load to aggregate results
    local_mem_per_sample: float =  local_gpu_memory / local_batch_size

    all_mems_per_sample: Tensor = torch.zeros(device_count, dtype=torch.float32, device=device) 
    all_mems_per_sample[rank] = torch.tensor(local_mem_per_sample, dtype=torch.float32, device=device)

    dist.all_reduce(all_mems_per_sample, op=dist.ReduceOp.SUM)

    smallest_mem_per_sample_index: int =  int(all_mems_per_sample.argmin())
    smallest_mem_per_sample: float = float(all_mems_per_sample[smallest_mem_per_sample_index].item())

    free_memory_per_gpu: List[float] = list(map(lambda s, m: float(m.item()) - (int(m.item() / s.item()) * smallest_mem_per_sample), all_mems_per_sample, all_local_gpu_memories))

    print("2")

    for r, m in enumerate(free_memory_per_gpu):
        if (smallest_mem_per_sample * (all_local_gpu_memories[r].item() / tot_gpu_memory) ) <= m:
            free_memory_per_gpu[r] -= smallest_mem_per_sample
            tot_batch_size += 1
            if r == rank:
                local_batch_size += 1

    # aggregating GPU always the one with the most space 
    main_gpu: int = free_memory_per_gpu.index(max(free_memory_per_gpu))

    run_name: str = f'{args.type}_cv_{args.shift}_oversample_{args.oversample}'
    checkpoint_filepath: str = f'{args.models_dir}/{run_name}.pth'

    torch.cuda.set_device(main_gpu)
    Logger.init(f'{run_name}_threshold_{threshold}', main_gpu)
    Logger.log(f'Tot Batch Size: {tot_batch_size}')
    Logger.log(f'Aggregate Results at GPU: {main_gpu}')
    Logger.log(f'Local Batch Size: {local_batch_size}', None)

    print("3")


    model_obj: DinoFeatureClassifier | StreamingNet | MLPHeader = create_model(model_type, device, clinical_data is not None)
	
    best_f1: float = float(0.0)
    best_loss: float = float(1.0)
    epoch: int = int(0)
    checkpoint_dict = None

    is_resume = os.path.isfile(checkpoint_filepath)

    print("4")

    if is_resume:
        Logger.log('Loading Stored')
        checkpoint_dict = torch.load(checkpoint_filepath)
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['model'].items()}
        model_obj.load_state_dict(model_state_dict)

        best_f1 = float(checkpoint_dict["best_f1"])
        best_loss = float(checkpoint_dict["best_loss"])
        epoch = int(checkpoint_dict["epoch"])

    model_obj = model_obj.to(device)

    if rank == main_gpu:
        if not args.t:
            wandb.init(
                    id=run_name,
                    project=args.project,
                    group=f'{args.type}',
                    name = f'{args.shift}',
                    config= {
                        "learning_rate": args.lr,
                        "architecture": f'{args.type}',
                        "dataset": "Tumor Budding Hotspots",
                        },
                    resume=is_resume
                    )
    assert model_obj is not None

    Logger.log(f'running distributed on {torch.cuda.device_count()} GPUs')

    model_dist: DistributedDataParallel = DistributedDataParallel(model_obj, device_ids=[device], output_device=device)
    model: SyncBatchNorm = SyncBatchNorm.convert_sync_batchnorm(model_dist)

    optimizer: Optimizer = Optimizer.new(model.parameters(), float(args.lr), int(args.epochs), checkpoint_dict)

    Logger.log("Loading Data ...")

    train_dataloader, val_dataloader, test_dataloader, class_weights = get_dataloaders(args.shift, args.data_dir, local_batch_size, args.oversample, model_type, args.i, args.p, clinical_data, header_only=model_type['type'] == 'header_only')

    weights = torch.from_numpy(class_weights).float().to(device)

    if not args.t:
        Logger.log(f'Class weights {class_weights}')

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)

    if args.t:
        Logger.log("Start Test ...")
    elif args.i:
        Logger.log("Start Training with smaller images ...")
    else:
        Logger.log("Fine Tuning for whole resolution")

    no_improvement = 0

    flag_tensor = torch.zeros(1).to(device)

    best_model_dict = None

    print("5")

    while True and not args.t:
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        train_loss, train_precision, train_recall, train_f1, _ = step(model, optimizer, criterion, train_dataloader, device, rank,  device_count, threshold=threshold)
        model.eval()
        val_dataloader.sampler.set_epoch(epoch)
        val_loss, val_precision, val_recall, val_f1, average_f1 = step(model, None, criterion, val_dataloader, device, rank, device_count, threshold=threshold, average=None)

        if rank == main_gpu:

            if val_loss is None or math.isnan(val_loss):
                val_loss = 1.0

            wandb.log({"train_loss": train_loss, "train_prec": train_precision, "train_recall": train_recall, "train_f1": train_f1, "val_loss": val_loss, "val_prec_0": val_precision[0], "val_prec_1": val_precision[1], "val_recall_0": val_recall[0], "val_recall_1": val_recall[1], "val_f1_0": val_f1[0], "val_f1_1": val_f1[1]})
            Logger.log(f'Epoch: {epoch}\nTrain:\nLoss: {train_loss}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}\nValidation:\nLoss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')

            assert average_f1 is not None and best_f1 is not None

            if best_model_dict is None or best_loss > val_loss or average_f1 > best_f1: 

                best_loss = val_loss if best_loss > val_loss else best_loss
                best_f1 = average_f1 if average_f1 > best_f1 else best_f1
                best_model_dict = model.state_dict()

                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= 80:
                    Logger.log("No Loss And F1 improvement for 40 Epoch, exiting training", None)
                    flag_tensor += 1

            model_path = f'{args.models_dir}/{run_name}.pth'
            torch.save({
                'epoch': epoch,
                'model': best_model_dict, 
                'optimizer': optimizer.optimizer.state_dict(),
                'scheduler': optimizer.scheduler.state_dict(),
                'best_f1': best_f1,
                'best_loss': best_loss,
                },
                model_path 
            )
            wandb.save(model_path)

        dist.all_reduce(flag_tensor)

        if flag_tensor == 1:
            break;
        epoch += 1

    # test values
    if args.t or not args.i:
        model = model.to(device)
        model.eval() 
        test_dataloader.sampler.set_epoch(epoch)
        test_loss, test_precision, test_recall, test_f1, _ = step(model, None, criterion, test_dataloader, device, rank, device_count, threshold=threshold, average=None)
        if rank == main_gpu:
            wandb.finish()
            Logger.log(f'Best Val loss = {best_loss}\nTest:\nLoss: {test_loss}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}', None)


if __name__ == "__main__":
    main()

