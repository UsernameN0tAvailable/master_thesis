#!/usr/bin/env python
# coding: utf-8

# # Emperical experiment for StreamingCNN

# To evaluate whether a neural network using streaming trains equivalently to the conventional training, we can train a CNN on small images using both methods, starting from the same initialization. We used a subset of the ImageNet dataset, [ImageNette](https://github.com/fastai/imagenette), using 100 examples of 10 ImageNet classes (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

import torch
import torchvision.transforms.functional


# # Model definition

class TopCNN(torch.nn.Module):
    def __init__(self):
        super(TopCNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x



"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def run(epochs, sCNN, net, trainloader, valloader, testvalloader, criterion, optimizer, scheduler):

    # sCNN uses hooks to correct gradients while streaming, 
    # so we have to disable it when training conventionally
    sCNN.enable()

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    no_improvements = 0
    
    for _ in range(epochs):
        running_loss = 0.0
        i = 0

        trues = []
        preds = []

        for images, labels in tqdm(trainloader):
            # first inference / stream through first network
            with torch.no_grad():
                first_output = sCNN.forward(images)
            first_output.requires_grad = True

            labels = labels.cuda()
            
            # inference final part of network
            second_output = net(first_output)

            # backpropagation through final network
            loss = criterion(second_output.double(), labels)
            loss.backward()

            # backpropagation through first network using checkpointing / streaming
            sCNN.backward(images, first_output.grad)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            preds.extend(torch.max(second_output.data, 1)[1].cpu().numpy())
            trues.extend(labels.cpu().numpy())

            running_loss += loss.item()
            i += 1

        train_loss = running_loss / i
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(trues, preds, average='weighted', zero_division=0.0)

        # validation

        val_trues = []
        val_preds = []

        val_loss = 0
        i = 0
        with torch.no_grad():
            for images, labels in valloader:
                first_output = sCNN.forward(images.cuda())
                second_output = net(first_output)

                loss = criterion(second_output.double(), labels.cuda())
                val_loss += loss.item()

                i += 1
                val_trues.extend(labels.cpu().numpy())
                val_preds.extend(torch.max(second_output.data, 1)[1].cpu().numpy())

        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_trues, val_preds, average=None, zero_division=0.0)

        val_loss = val_loss / i
            
        wandb.log({"train_loss": train_loss, "train_prec": train_precision, "train_recall": train_recall, "train_f1": train_f1, "val_loss": val_loss, "val_prec_0": val_precision[0], "val_prec_1": val_precision[1], "val_recall_0": val_recall[0], "val_recall_1": val_recall[1], "val_f1_0": val_f1[0], "val_f1_1": val_f1[1]})

        average_val_f1 = ((val_f1[0] + val_f1[1]) / 2.0)

        if val_loss < best_val_loss or average_val_f1 > best_val_f1:
            best_val_f1 = average_val_f1 if average_val_f1 > best_val_f1 else best_val_f1
            best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
            no_improvements = 0

        if no_improvements >= 40:
            print(f'Train completed Best F1: {best_val_f1}, best Val Loss: {best_val_loss}')
            wandb.finish()
            break

    test_trues = []
    test_preds = []

    test_loss = 0
    i = 0
    with torch.no_grad():
        for images, labels in testvalloader:
            first_output = sCNN.forward(images.cuda())
            second_output = net(first_output)

            loss = criterion(second_output.double(), labels.cuda())
            test_loss += loss.item()

            i += 1
            test_trues.extend(labels.cpu().numpy())
            test_preds.extend(torch.max(second_output.data, 1)[1].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(test_trues, test_preds, average=None, zero_division=0.0)

    test_loss = test_loss / i
            
    print(f'Test:\nLoss: {test_loss}, Precision:{precision}, Recall: {recall}, F1: {f1}')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--shift", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tile_size", type=lambda c: int(c) if int(c) > 0 and int(c) % 16 == 0 else argparse.ArgumentTypeError(f"{c} crop size has to be multiple of 16"))
    parser.add_argument("--batch_size", type=lambda x: int(x) if int(x) > 0 else argparse.ArgumentTypeError(f"{x} is an invalid batch size"))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--type", type=str, default="cnn", choices=["cnn", "vit", "resnet"])
    parser.add_argument("--img_size", type=int, default=3600)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--freeze", type=int, default=1)

    args = parser.parse_args()

    image_size = args.img_size

    is_frozen = True if args.freeze == 1 else False

    wandb.init(
        project=f'pT1',
        group=f'Stream {args.type} Tile Size: {args.tile_size} Batch Size: {args.batch_size} Frozen {is_frozen}',
        name = f'cv{args.shift}',
        config= {
            "learning_rate": args.lr,
            "architecture": "sCNN",
            "dataset": "Tumor Budding Hotspots",
             }
        )

    torch.set_printoptions(precision=10)
    torch.manual_seed(0)


    # to compare the networks we want to train deterministically
    torch.backends.cudnn.deterministic = True

    bottom_net = None

    if args.type == "cnn":
        bottom_net = BottomCNN().cuda()
    elif args.type == "vit":
        bottom_net = BottomVit().cuda()
    else:
        bottom_net = BottomResNet().cuda()


    top_net = TopCNN().cuda()

    params_count = count_parameters(bottom_net) + count_parameters(top_net)
    print("params count", params_count)

    # # Configure streamingSGD
    sCNN = StreamingCNN(bottom_net, tile_shape=(1, 3, args.tile_size, args.tile_size), deterministic=True, verbose=False)

    train_dataloader, val_dataloader, test_dataloader, weights = get_dataloaders(args.shift, args.data_dir,  image_size, args.batch_size, 0.5, True, image_size)

    weights = torch.from_numpy(weights).to("cuda")

    criterion = torch.nn.CrossEntropyLoss(weight= weights)

    for mod in top_net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
            mod.bias.data.fill_(0)

    original_initialization_stream_net = copy.deepcopy(bottom_net.state_dict())
    original_initialization_net = copy.deepcopy(top_net.state_dict())

    params = list(bottom_net.parameters()) + list(top_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
 
    torch.backends.cudnn.deterministic = True

    # load original state
    bottom_net.load_state_dict(original_initialization_stream_net)
    top_net.load_state_dict(original_initialization_net)


    # initialize the dataloaders after re-seed, if we do this in both training cases the order is identical

    run(args.epochs, sCNN, top_net, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler)


if __name__ == '__main__':
    main()
"""
