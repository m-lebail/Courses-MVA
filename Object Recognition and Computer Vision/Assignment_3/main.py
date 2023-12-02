import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import wandb
from tqdm import tqdm


import torchvision.models as models


from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--unfreeze",
        type=float,
        default=4,
        metavar="UF",
        help="Epoch to start unfreezing (default: 4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        metavar="WD",
        help="weight decay (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--name_wb_project",
        type=str,
        default="default",
        metavar="WANB",
        help="name of the wandb project",
    )
    
    
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    metrics = {"train_loss" : loss.data.item(), "train_accuracy" : 100.0 * correct / len(train_loader.dataset)}
    wandb.log({**metrics})


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    metrics_val = {"val_loss" : validation_loss, "val_accuracy" : 100.0 * correct / len(val_loader.dataset)}
    wandb.log({**metrics_val})
    return validation_loss


def main(config=None):
    
    
    """Default Main Function."""
    # options
    args_ = opts()

    wandb.init(
        project=args_.name_wb_project,
        config={"learning_rate" : args_.lr ,
                "batch_size ": args_.batch_size,
                "epochs ": args_.epochs,
                "momentum" : args_.momentum,
                "unfreeze" : args_.unfreeze,
                "weight_decay": args_.weight_decay
                }
    )
    #with wandb.init(config=config):

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args_.seed)

    # Create experiment folder
    if not os.path.isdir(args_.experiment):
        os.makedirs(args_.experiment)

    # load model and transform
    model, data_transforms = ModelFactory(args_.model_name).get_all()
    print(model)
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    original_dataset = datasets.ImageFolder(args_.data + "/train_images", transform=data_transforms[0])
    #Data augmentation
    rotation_dataset = datasets.ImageFolder(args_.data + "/train_images", transform=data_transforms[1])
    # Concatenate datasets
    merged_dataset = torch.utils.data.ConcatDataset([original_dataset, rotation_dataset])
    

    train_loader = torch.utils.data.DataLoader(
        #datasets.ImageFolder(args_.data + "/train_images", transform=data_transforms),
        merged_dataset,
        batch_size=args_.batch_size,
        shuffle=True,
        num_workers=args_.num_workers,
    )
   
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args_.data + "/val_images", transform=data_transforms[0]),
        batch_size=args_.batch_size,
        shuffle=False,
        num_workers=args_.num_workers,
    )

    # Setup optimizer
    if args_.model_name != "basic_cnn":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args_.lr, momentum=args_.momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args_.lr, momentum=args_.momentum)

    

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args_.epochs + 1):
        
        if epoch == args_.unfreeze:
            for param in model.parameters():
               param.requires_grad = True
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args_.lr/10, momentum=args_.momentum, weight_decay=args_.weight_decay)
            
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args_)
    
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args_.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = args_.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args_.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
     

    wandb.finish()

if __name__ == "__main__":
    main()
    
