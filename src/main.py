import os
import argparse
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.dino import DinoFeatureClassifier
from models.streaming.top import TopCNN
from models.streaming.bottom import BottomCNN
from models.streaming.scnn import StreamingNet

import re

# distribute
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import wandb

import pynvml

from pipeline_utils import get_dataloaders, Logger

from torch.nn import SyncBatchNorm


def step(model, optimizer, scheduler, criterion, dataloader, device, rank, device_count, local_batch_size: int, tot_batch_size: int, average="weighted"):

    running_loss = torch.tensor(0.0, device=device)
    true = []
    preds = []

    for images, labels in dataloader:
        labels = labels.to(device)
        images = images.to(device)

        if optimizer is not None: optimizer.zero_grad()

        output, loss = model.module.step(images, labels, criterion, optimizer is not None)
        running_loss += loss.item()
        predicted = torch.max(output.data, 1)[1]
        
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

    if optimizer is not None:
        optimizer.step()
        scheduler.step()

    print("\nbef ", rank, running_loss)

    running_loss /= len(dataloader) 
    running_loss *= local_batch_size

    print("aft ", rank, running_loss)

    dist.reduce(running_loss, dst=0, op=dist.ReduceOp.AVG)

    print("aft reduce ", rank, running_loss)

    epoch_loss = running_loss.item() / tot_batch_size

    true_tensor = torch.tensor(true, device=device)
    preds_tensor = torch.tensor(preds, device=device)

    gathered_true_tensors = [torch.zeros_like(true_tensor) for _ in range(device_count)]
    gathered_preds_tensors = [torch.zeros_like(preds_tensor) for _ in range(device_count)]

    dist.all_gather(gathered_true_tensors, true_tensor)
    dist.all_gather(gathered_preds_tensors, preds_tensor)

    if rank == 0:
        gathered_trues = torch.cat(gathered_true_tensors, dim=0).cpu().numpy()
        gathered_preds = torch.cat(gathered_preds_tensors, dim=0).cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(gathered_trues, gathered_preds, average=average, zero_division=0.0)
        return epoch_loss, precision, recall, f1

    return None, None, None, None


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
    stream_match = re.match(r'^stream\[(vit|cnn|resnet|unet)->(vit|cnn|resnet|unet), (\d+)\]$', s)
    if stream_match:
        top_m = stream_match.group(1)
        bottom_m = stream_match.group(2)
        n = int(stream_match.group(3))

        if n <= 0:
            raise ValueError(f"Invalid value for 'n': {n}. It must be a positive integer.")
        
        if (top_m == "vit" or bottom_m == "vit") and n % 16 != 0:
            raise ValueError(f"Invalid combination: stream[{top_m}|{bottom_m}, {n}]. When 'm' is 'vit', 'n' must be divisible by 16.")
        
        return {'type': 'stream', 'top': top_m, 'bottom': bottom_m, 'value': n}

    raise ValueError("Model type not available\nPossible types:\n- vit[<tile_size>]\n- stream[<cnn | vit | resnet | unet>|<cnn | vit | resnet | unet>, <patch_size>]")


def create_model(param, lr: float, epochs: int, load_path: str, device: str):


    if param['type'] == 'vit':
        model = DinoFeatureClassifier()
        if os.path.isfile(load_path):
            Logger.log('Loading stored ViT')
            checkpoint = torch.load(load_path)

            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}

            model.load_state_dict(model_state_dict)

            optimizer = AdamW(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer'])

            scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
            scheduler.load_state_dict(checkpoint['scheduler'])

            return model, optimizer, scheduler, float(checkpoint['best_f1']), float(checkpoint['best_loss']), int(checkpoint['epoch']), True
        else:
            Logger.log('Creating New ViT')
            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
            return model, optimizer, scheduler, 0.0, float('inf'), 0, False
    elif param['type'] == 'stream':

        top_param = param['top']
        bottom_param = param['bottom']

        if top_param == 'cnn':
            top_net = TopCNN()
        elif top_param == 'vit':
            top_net = TopCNN()
        elif top_param == 'resnet':
            top_net = TopCNN()
        elif top_param == 'unet':
            top_net = TopCNN()
        else:
            raise ValueError(f'No {top_param} top net available!!')

        if bottom_param == 'cnn':
            bottom_net = BottomCNN()
        elif bottom_param == 'vit':
            bottom_net = BottomCNN()
        elif bottom_param == 'resnet':
            bottom_net = BottomCNN()
        elif bottom_param == 'unet':
            bottom_net = BottomCNN()
        else:
            raise ValueError(f'No {bottom_param} bottom net available!!')


        if os.path.isfile(load_path):
            Logger.log(f'Loading stored {param}')
            checkpoint = torch.load(load_path)
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}

            model = StreamingNet(top_net.to(device), bottom_net.to(device), param['value']) 
            model.load_state_dict(model_state_dict)

            optimizer = AdamW(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer'])

            scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
            scheduler.load_state_dict(checkpoint['scheduler'])

            return model, optimizer, scheduler, float(checkpoint['best_f1']), float(checkpoint['best_loss']), int(checkpoint['epoch']), True
        else:
            Logger.log(f'Creating New {param}')
            model = StreamingNet(top_net.to(device), bottom_net.to(device), param['value'])
            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
            return model, optimizer, scheduler, 0.0, float('inf'), 0, False
    else:
        raise ValueError(f'{param} Not Implemented')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--type", type=str, default="vit[2048]")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--batch_size", type=check_batch_size, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--oversample", type=float, default=0.5)
    parser.add_argument("--test_only", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    model_type = validate_model_and_extract(args.type)
    test_only = args.test_only == 1


    dist.init_process_group(backend='nccl', init_method='env://')

    rank = int(os.environ["LOCAL_RANK"])

    device_count = torch.cuda.device_count()
    device = torch.device(f'cuda:{rank}') if args.device == 'cuda' else torch.device('cpu')


    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    local_gpu_memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 2))
    pynvml.nvmlShutdown()

    tot_gpu_memory = torch.tensor(local_gpu_memory, dtype=torch.int, device=device)
    dist.all_reduce(tot_gpu_memory, op=dist.ReduceOp.SUM)

    local_batch_size = int(args.batch_size * (local_gpu_memory / tot_gpu_memory))
    tot_batch_size = torch.tensor(local_batch_size, dtype=torch.int, device=device)
    dist.all_reduce(tot_batch_size, op=dist.ReduceOp.SUM)
    tot_batch_size = tot_batch_size.item()

    run_name = f'{args.type}_shift_{args.shift}_batch_size_{tot_batch_size}_oversample_{args.oversample}'
    checkpoint_filepath = f'{args.models_dir}/{run_name}.pth'


    Logger.init(run_name)
    Logger.log(f'Tot Batch Size: {tot_batch_size}')
    Logger.log(f'Using device: {device} with batch size: {local_batch_size}', None)


    model, optimizer, scheduler, best_f1, best_loss, epoch, is_resume =  create_model(model_type, float(args.lr), int(args.epochs), checkpoint_filepath, device)

    model = model.to(device)


    if rank == 0:
        if not test_only:
            wandb.init(
                    project=f'pT1 Test',
                    group=f'{args.type}',
                    name = f'cv{args.shift}',
                    config= {
                        "learning_rate": args.lr,
                        "architecture": f'{args.type}',
                        "dataset": "Tumor Budding Hotspots",
                        },
                    resume=is_resume
                    )
    assert model is not None

    Logger.log(f'running distributed on {torch.cuda.device_count()} GPUs')

    model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) 

    Logger.log("Loading Data ...")

    train_dataloader, val_dataloader, test_dataloader, class_weights = get_dataloaders(args.shift, args.data_dir, local_batch_size, args.oversample, model_type)

    weights = torch.from_numpy(class_weights).float().to(device)

    Logger.log(f'Class weights {class_weights}')

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)

    Logger.log("Start Training ...")

    no_improvement = 0

    flag_tensor = torch.zeros(1).to(device)

    epochs = 0 if test_only else 10000 

    while epoch < epochs:
        model.train()
        train_loss, train_precision, train_recall, train_f1 = step(model, optimizer, scheduler, criterion, train_dataloader, device, rank, device_count, local_batch_size, tot_batch_size)
        model.eval()
        val_loss, val_precision, val_recall, val_f1 = step(model, None, None, criterion, val_dataloader, device, rank, device_count, local_batch_size, tot_batch_size, average=None)
        if rank == 0:
            wandb.log({"train_loss": train_loss, "train_prec": train_precision, "train_recall": train_recall, "train_f1": train_f1, "val_loss": val_loss, "val_prec_0": val_precision[0], "val_prec_1": val_precision[1], "val_recall_0": val_recall[0], "val_recall_1": val_recall[1], "val_f1_0": val_f1[0], "val_f1_1": val_f1[1]})
            Logger.log(f'Epoch: {epoch + 1}\nTrain:\nLoss: {train_loss}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}\nValidation:\nLoss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}', None)

            average_f1 = (val_f1[0] + val_f1[1]) / 2

            if  best_loss > val_loss or average_f1 > best_f1: 

                best_loss = val_loss if best_loss > val_loss else best_loss
                best_f1 = average_f1 if average_f1 > best_f1 else best_f1

                no_improvement = 0
                model_path = f'{args.models_dir}/{run_name}.pth'
                Logger.log(f'Saving model to {model_path}', None)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'best_loss': best_loss,
                    },
                    model_path 
                    )
                wandb.save(model_path)
            else:
                no_improvement += 1
                if no_improvement >= 40:
                    Logger.log("No Loss And F1 improvement for 40 Epoch, exiting training", None)
                    flag_tensor += 1

        dist.all_reduce(flag_tensor)

        if flag_tensor == 1:
            break;
        epoch += 1

    # test values
    model = model.to(device)
    model.eval() 
    test_loss, test_precision, test_recall, test_f1 = step(model, None, None, criterion, test_dataloader, device, rank, device_count, args.t, average=None)
    if rank == 0:
        wandb.finish()
        Logger.log(f'Best Val loss = {best_loss}\nTest:\nLoss: {test_loss}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}', None)


if __name__ == "__main__":
    main()

