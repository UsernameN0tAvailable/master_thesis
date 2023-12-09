import os
import shutil
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
from pipeline_utils import Rank, get_dataloaders, RunTime, Optimizer
from torch.nn import SyncBatchNorm

import torchvision.transforms as transforms

from PIL import Image

# fix ssl pytorch bug
ssl._create_default_https_context = ssl._create_unverified_context


def store_maps(labels: Tensor, predictions: Tensor, names: List[str], maps: Tensor, path: str, overlay_dir: Optional[str] = None) -> None:
        maps_path = f'{path}'
        if not os.path.exists(maps_path): os.makedirs(maps_path, exist_ok=True) 

        true_pos_path = f'{maps_path}/true_pos'
        if not os.path.exists(true_pos_path): os.makedirs(true_pos_path)

        false_pos_path = f'{maps_path}/false_pos'
        if not os.path.exists(false_pos_path): os.makedirs(false_pos_path)

        true_neg_path = f'{maps_path}/true_neg'
        if not os.path.exists(true_neg_path): os.makedirs(true_neg_path)

        false_neg_path = f'{maps_path}/false_neg'
        if not os.path.exists(false_neg_path): os.makedirs(false_neg_path)

            
        for i, map_tensor in enumerate(maps):

            prediction = int(predictions[i].int())
            true_label = int(labels[i].int())

            if prediction == true_label:
                if prediction == 1:
                    store_path = true_pos_path
                else:
                    store_path = true_neg_path
            else:
                if prediction == 1:
                    store_path = false_pos_path
                else:
                    store_path = false_neg_path



            if overlay_dir is not None:
                _, height, _ = map_tensor.shape
                original_image = Image.open(f'{overlay_dir}/hotspots-png/{names[i]}.png')
                original_image = transforms.Compose([transforms.CenterCrop(height), transforms.ToTensor()])(original_image)
                green_channel_tensor = torch.zeros_like(map_tensor)
                green_channel_tensor[1] = map_tensor[1] * 10.0
                map_tensor = original_image - green_channel_tensor 

            
            image = transforms.ToPILImage()(map_tensor)
            image.save(f'{store_path}/{names[i]}_map.png')


def step(model, optimizer: Optional[Optimizer], criterion, dataloader, dirs: Optional[Tuple[str, Optional[str], Optional[str]]] = None):

    device = RunTime.device()
    device_count = RunTime.device_count

    running_loss: Tensor = torch.tensor(0.0, device=device)
    true: List[np.ndarray] = []
    preds: List[np.ndarray] = []

    data_dir = None
    feature_maps_dir = None
    activation_maps_dir = None

    if dirs is not None:
        data_dir, feature_maps_dir, activation_maps_dir = dirs

    store_feature_maps: bool = feature_maps_dir is not None
    store_activation_maps: bool = activation_maps_dir is not None

    for images, labels, clinical_data, img_names  in dataloader:
        labels: Tensor = labels.to(device)
        images: Tensor = images.to(device)
        clinical_data: Tensor = clinical_data.to(device)

        if optimizer is not None: optimizer.optimizer.zero_grad()

        output, loss = model.module.step(images, labels, criterion, optimizer, clinical_data, store_feature_maps=store_feature_maps, store_activation_maps=store_activation_maps)
        running_loss += loss.item() * images.size(0)
        predicted = torch.max(output.data, 1)[1]
        
        true.extend(labels.cpu().numpy())
        preds.extend(predicted.cpu().numpy())

        if optimizer is not None:
            optimizer.optimizer.step()
            optimizer.scheduler.step()
            optimizer.optimizer.zero_grad() 
        # store all activation maps
        elif store_feature_maps:
            store_maps(labels, predicted, img_names, model.module.get_feature_maps(), feature_maps_dir)

        elif store_activation_maps:
            store_maps(labels, predicted, img_names, model.module.get_activation_maps(), activation_maps_dir, data_dir)

    dist.all_reduce(running_loss, op=dist.ReduceOp.AVG) # type: ignore
    epoch_loss: float = running_loss.item() / len(dataloader.dataset)

    true_tensor: Tensor = torch.tensor(true, device=device)
    preds_tensor: Tensor = torch.tensor(preds, device=device)

    gathered_true_tensors: List[Tensor] = [torch.zeros_like(true_tensor) for _ in range(device_count)]
    gathered_preds_tensors: List[Tensor] = [torch.zeros_like(preds_tensor) for _ in range(device_count)]

    dist.all_gather(gathered_true_tensors, true_tensor)
    dist.all_gather(gathered_preds_tensors, preds_tensor)

    return DistributedResult(gathered_true_tensors, gathered_preds_tensors, epoch_loss) 


class DistributedResult:

    def  __init__(self, truths: List[Tensor], predictions: List[Tensor], loss: float) -> None:
        self.truths: List[Tensor] = truths
        self.predictions: List[Tensor] = predictions
        self.loss: float = loss

    def compute(self, average: Optional[str] = "weighted"):
        gathered_trues = torch.cat(self.truths, dim=0).cpu().numpy()
        gathered_preds = torch.cat(self.predictions, dim=0).cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(gathered_trues, gathered_preds, average=average, zero_division=0.0) # type: ignore

        return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_f1': f1_score(gathered_trues, gathered_preds, average="weighted", zero_division=0.0), # type: ignore
                'loss': self.loss
        }

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


def create_model(param: Dict[str, Any], device: str, store_activation_maps: bool = False) -> DinoFeatureClassifier | StreamingNet | MLPHeader:

    RunTime.log(f"Model: {param}")

    if param['type'] == 'vit':
        return DinoFeatureClassifier()
    elif param['type'] == 'stream':

        top_param = param['top']
        bottom_param = param['bottom']

        if top_param == 'cnn':
            top_net = TopCNN()
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

        return StreamingNet(top_net.to(device), bottom_net.to(device), param['value'], store_activation_maps) 
    elif param['type'] == 'header_only':
        return MLPHeader()
    else:
        raise ValueError(f'{param} Not Implemented')


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=80)
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
    parser.add_argument("--clinical_data_dir", type=str, default=None, help="Additional clinical data file path for the classification net")
    parser.add_argument("--feature_maps_dir", type=str, default=None, help="If defined, stores feature maps into the dir")
    parser.add_argument("--activation_maps_dir", type=str, default=None, help="If defined, stores activation maps into the dir")

    args = parser.parse_args()

    if (args.feature_maps_dir is not None or args.activation_maps_dir is not None ) and not args.t:
        raise ValueError("Cannot store maps if not on testing phase!")

    clinical_data: Optional[Dict[str, np.ndarray]] = None;

    if args.clinical_data_dir is not None:
        clinical_data = load_clinical_data(args.clinical_data_dir)

    if args.t and args.i:
        raise ValueError("Cannot use reduced image size for testing!!")

    model_type = validate_model_and_extract(args.type)

    if model_type['type'] == 'header_only' and clinical_data is None:
        raise ValueError("Cannot Execture with header only without clinical data")

    dist.init_process_group(backend='nccl', init_method='env://')

    rank: int = int(os.environ["LOCAL_RANK"])

    def create_or_replace_folder(path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    if args.feature_maps_dir is not None:
        RunTime.execute(create_or_replace_folder, args.feature_maps_dir)

    if args.activation_maps_dir is not None:
        RunTime.execute(create_or_replace_folder, args.activation_maps_dir)

    device_count: int = torch.cuda.device_count()
    device: str = str(torch.device(f'cuda:{rank}') if args.device == 'cuda' else torch.device('cpu'))

    # Memory Load Management
    local_gpu_memory: int = int(torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2))

    all_local_gpu_memories: Tensor = torch.zeros(device_count, dtype=torch.int64, device=device) 
    all_local_gpu_memories[rank] = torch.tensor(local_gpu_memory, dtype=torch.int64, device=device)

    dist.all_reduce(all_local_gpu_memories, op=dist.ReduceOp.SUM)
    tot_gpu_memory: Number = all_local_gpu_memories.sum().item()

    local_batch_size: int = int(args.batch_size * (local_gpu_memory / tot_gpu_memory))
 
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
    RunTime.init(run_name, main_gpu, device_count, args.device)
    RunTime.log(f'Tot Batch Size: {tot_batch_size}')
    RunTime.log(f'Aggregate Results at GPU: {main_gpu}')
    RunTime.log(f'Local Batch Size: {local_batch_size}', Rank.Local)


    model_obj: DinoFeatureClassifier | StreamingNet | MLPHeader = create_model(model_type, device, args.activation_maps_dir is not None)
	
    best_f1: float = float(0.0)
    best_loss: float = float(1.0)
    epoch: int = int(0)
    checkpoint_dict = None

    is_resume = os.path.isfile(checkpoint_filepath)

    if is_resume:
        RunTime.log('Loading Stored')
        checkpoint_dict = torch.load(checkpoint_filepath)
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['model'].items()}
        model_obj.load_state_dict(model_state_dict)

        best_f1 = float(checkpoint_dict["best_f1"])
        best_loss = float(checkpoint_dict["best_loss"])
        epoch = int(checkpoint_dict["epoch"])

    model_obj = model_obj.to(device)

    if not args.t:
        RunTime.execute(
                wandb.init,
                id=run_name, 
                project=args.project, 
                group=f'{args.type}', 
                name = f'{args.shift}', 
                config= {
                    "learning_rate": args.lr, 
                    "architecture": f'{args.type}', 
                    "dataset": "Tumor Budding Hotspots"
                },
                resume=is_resume
             )
 
    assert model_obj is not None

    RunTime.log(f'running distributed on {torch.cuda.device_count()} GPUs')

    model_dist: DistributedDataParallel = DistributedDataParallel(model_obj, device_ids=[device], output_device=device)
    model: SyncBatchNorm = SyncBatchNorm.convert_sync_batchnorm(model_dist)

    optimizer: Optimizer = Optimizer.new(model.parameters(), float(args.lr), int(args.epochs), checkpoint_dict)

    RunTime.log("Loading Data ...")

    train_dataloader, val_dataloader, test_dataloader, class_weights = get_dataloaders(args.shift, args.data_dir, local_batch_size, args.oversample, model_type, args.i, args.p, clinical_data, header_only=model_type['type'] == 'header_only')

    weights = torch.from_numpy(class_weights).float().to(device)

    if not args.t:
        RunTime.log(f'Class weights {class_weights}')

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)

    if args.t:
        RunTime.log("Start Test ...")
    elif args.i:
        RunTime.log("Start Training with smaller images ...")
    else:
        RunTime.log("Fine Tuning for whole resolution")

    no_improvement = 0

    flag_tensor = torch.zeros(1).to(device)

    best_model_dict = None

    while True and not args.t:
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        train_result = step(model, optimizer, criterion, train_dataloader)
        model.eval()
        val_dataloader.sampler.set_epoch(epoch)
        validation_result = step(model, None, criterion, val_dataloader)

        def epoch_step(model, train_result: DistributedResult, val_result: DistributedResult, best_model_dict, best_loss: float, best_f1: float, no_improvement: int, flag_tensor: Tensor) -> None:

            t_result = train_result.compute('weighted')
            v_result = val_result.compute(None)

            if v_result['loss'] is None or math.isnan(v_result['loss']):
                v_result['loss'] = 1.0

            train_loss = t_result['loss']
            train_precision = t_result['precision']
            train_recall = t_result['recall']
            train_f1 = t_result['f1']

            val_loss = v_result['loss']
            val_precision = v_result['precision']
            val_recall = v_result['recall']
            val_f1 = v_result['f1']


            wandb.log({"train_loss": t_result['loss'], "train_prec": t_result['precision'], "train_recall": t_result['recall'], "train_f1": t_result['f1'], "val_loss": v_result['loss'], "val_prec_0": v_result['precision'][0], "val_prec_1": v_result['precision'][1], "val_recall_0": v_result['recall'][0], "val_recall_1": v_result['recall'][1], "val_f1_0": v_result['f1'][0], "val_f1_1": v_result['f1'][1]})
            RunTime.log(f'Epoch: {epoch}\nTrain:\nLoss: {train_loss}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}\nValidation:\nLoss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')

            assert v_result['avg_f1'] is not None and best_f1 is not None

            if best_model_dict is None or best_loss > val_loss or v_result['avg_f1'] > best_f1: 

                best_loss = val_loss if best_loss > val_loss else best_loss
                best_f1 = v_result['avg_f1'] if v_result['avg_f1'] > best_f1 else best_f1
                best_model_dict = model.state_dict()

                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= args.epochs:
                    RunTime.log(f'No Loss And F1 improvement for {args.epochs} Epochs, exiting training')
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

        RunTime.execute(epoch_step, model, train_result, validation_result, best_model_dict, best_loss, best_f1, no_improvement, flag_tensor)

        dist.all_reduce(flag_tensor)

        if flag_tensor == 1:
            break;
        epoch += 1

    # test values
    if args.t or not args.i:

        def test_eval(result: DistributedResult) -> None:
            wandb.finish()
            test_results = result.compute(average=None)
            precision = test_results['precision']
            loss = test_results['loss']
            recall = test_results['recall']
            f1 = test_results['f1']
            RunTime.log(f'Best Val loss = {best_loss}\nTest:\nLoss: {loss}, Precision: {precision}, Recall: {recall}, F1: {f1}')

        model = model.to(RunTime.device())
        model.eval() 
        test_dataloader.sampler.set_epoch(epoch)
        test_result = step(model, None, criterion, test_dataloader, dirs=(args.data_dir, args.feature_maps_dir, args.activation_maps_dir))

        RunTime.execute(test_eval, test_result)


if __name__ == "__main__":
    main()

