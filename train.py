import argparse
import datetime
from pyexpat import model
import torch
import os
import sys
from torch.utils.data import DataLoader
import wandb
import time

from Dataset import Image2NodeDataset
from models import Image2NodeNet, ResnetEncoder
from loss import losses_node
from utils import get_validation_metric

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help='starting epoch')
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs to train the model")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizers")
parser.add_argument("--output_path", type=str, default='checkpoints', help="Path to save checkpoints")
parser.add_argument("--exp", type=str, default='1', help='experiment identifier')
parser.add_argument("--wt_decay", type=float, default=0.0, help="Weight decay hyperparameter")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="model checkpoint inerval")

#Model related arguments
parser.add_argument("--in_channels", type=int, default=3, help="number of input image channels")
parser.add_argument('-a', '--arch_name', type=str, default='resnet18', help='architecture of resnet encoder')
parser.add_argument("--pretrained", action="store_true", help="use pretrained resnet model?")

parser.add_argument("--hd_sz", type=int, default=256, help="Hidden representation size of RNN")
parser.add_argument("--unq_op", type=int, default=6, help="number of unique operations/nodes")


opt = parser.parse_args()

opt.cuda = torch.cuda.is_available()
os.makedirs(os.path.join(opt.output_path, opt.exp), exist_ok=True)
os.makedirs(os.path.join(opt.output_path, opt.exp, 'saved_models'), exist_ok=True)

#Initialize Dataset and dataloaders
train_dataset = Image2NodeDataset('./data/train.json', './data/complete_index_sequence.json')
dev_dataset = Image2NodeDataset('./data/dev.json', './data/complete_index_sequence.json')

train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False)

#Intialize model
im_encoder = ResnetEncoder(arch_name=opt.arch_name, in_channels=opt.in_channels, 
                           pretrained=opt.pretrained)
image2node_net = Image2NodeNet(hd_sz=opt.hd_sz, input_size=512, inp_op_sz=opt.unq_op+2,
                             encoder=im_encoder)
if opt.cuda:
    im_encoder.cuda()
    image2node_net.cuda()

optimizer = torch.optim.Adam(
    [para for para in image2node_net.parameters() if para.requires_grad],
    weight_decay=opt.weight_decay,
    lr=opt.lr)

runs = wandb.init(project="Image2Node", entity = "shrawan", name=f"{opt.exp}", reinit=True)
prev_time = time.time()
best_val_loss = float('inf')
for epoch in range(opt.epoch, opt.n_epochs):
    image2node_net.train()
    for i, batch in enumerate(train_loader):
        image = batch['image']
        input_op_idx, label, program_lens = batch['inp_op'], batch['label'], batch['program_len']
        # Reshaping and getting one hot encoding of input operations
        input_op = torch.zeros((input_op_idx.shape[0], input_op_idx.shape[1], opt.unq_op+2))
        input_op = input_op.scatter_(2, input_op_idx.unsqueeze(2), 1)

        if opt.cuda:
            image = image.cuda()
            input_op = input_op.cuda()
            program_len = program_lens[-1].cuda()

        optimizer.zero_grad()
        output = image2node_net([image, input_op, program_len])
        loss = losses_node(out=output, labels=label, time_steps=program_len+1)
        loss.backward()
        optimizer.step()

        batches_done = epoch * len(train_loader) + i
        batches_left = opt.n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] Training loss: %f ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader),
                loss.item(),
                time_left
            )
        )

 
        wandb.log({
            "Training CE loss": loss.item(),
        })

    val_loss = get_validation_metric(image2node_net, dev_loader, opt)
    wandb.log({
        "validation CE loss": val_loss
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Best val_loss: %f" % val_loss)
        print("Saving best model for epoch %d" % epoch)
        torch.save({
            'epoch': epoch,
            'model_state': image2node_net.state_dict(),
            'best_val_loss': best_val_loss 
        }, "%s/%s/saved_models/best_model.pth" % (opt.output_path, opt.exp))


    # Save model checkpoints
    if epoch % opt.checkpoint_interval == 0:
        torch.save(image2node_net.state_dict(), "%s/%s/saved_models/image2node_net_%d.pth" % (opt.output_path, opt.exp, epoch))
        torch.save(optimizer.state_dict(), "%s/%s/saved_models/Optimizer_%d.pth" % (opt.output_path, opt.exp, epoch))






