import os
import sys
import torch
import argparse
from tqdm import tqdm
from models.mabdt import Model as EMPFNet
import torch.optim as optim
import torch.nn.functional as F
from losses.perceptual import LossNetwork
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.utils import torchPSNR, setseed
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets.datasets import MyTrainDataSet, MyValueDataSet
import math
from utils.utils import expand2square
from utils.model_utils import network_parameters, load_checkpoint, load_start_epoch, load_best_metrics, load_optim
from losses.Gradient_Loss import Gradient_Loss
def train(args):

    cudnn.benchmark = True
    setseed(args.seed)
    Gradient_loss = Gradient_Loss(device="cuda")
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model if args.cuda else vgg_model
    vgg_model=vgg_model.to("cuda" if torch.cuda.is_available() else "cpu")
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    # model
    model_restoration = EMPFNet().cuda() if args.cuda else EMPFNet()
    # optimizer
    new_lr = args.lr
    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr_min)
    # training dataset
    path_train_input, path_train_target = args.train_data + '/hazy/', args.train_data + '/clear/'
    datasetTrain = MyTrainDataSet(path_train_input, path_train_target, patch_size=args.patch_size_train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=args.batch_size_train, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    # validation dataset
    path_val_input, path_val_target = args.val_data + '/hazy/', args.val_data + '/clear/'
    datasetValue = MyValueDataSet(path_val_input, path_val_target, patch_size=args.patch_size_val)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=args.batch_size_val, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    # load pre model
    # if os.path.exists(args.save_state):
    #     if args.cuda:
    #         model_restoration.load_state_dict(torch.load(args.save_state))
    #     else:
    #         model_restoration.load_state_dict(torch.load(args.save_state, map_location=torch.device('cpu')))
    #Resume
    best_psnr = 0
    start_epoch = 0
    if args.resume:
        load_checkpoint(model_restoration, args.resume_state)
        start_epoch = load_start_epoch(args.resume_state) + 1
        best_psnr = load_best_metrics(args.resume_state)
        load_optim(optimizer, args.resume_state)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
    scaler = GradScaler() #混合精度训练
    
    for epoch in range(start_epoch,args.epoch):
        model_restoration.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        # train
        for index, (x, y) in enumerate(iters, 0):

            model_restoration.zero_grad()
            optimizer.zero_grad()

            input_train = Variable(x).cuda() if args.cuda else Variable(x)
            target_train = Variable(y).cuda() if args.cuda else Variable(y)
            input_train, mask = expand2square(input_train.cuda(), factor=128) 
            # 1×3×512×512
            with autocast(args.autocast):
                if args.only_last:
                    restored_train = model_restoration(input_train, only_last=args.only_last, mask=1-mask)
                    loss = 1 * F.l1_loss(restored_train, target_train) + 0.5 * loss_network(restored_train, target_train) + 1 * Gradient_loss(restored_train, target_train)
                else:
                    restored_train, fake_image_x8,fake_image_x4, fake_image_x2 = model_restoration(input_train, only_last=args.only_last, mask=1-mask)
                    fake_image_x8 = fake_image_x8.to(dtype=torch.float32)
                    fake_image_x4 = fake_image_x4.to(dtype=torch.float32)
                    fake_image_x2 = fake_image_x2.to(dtype=torch.float32)
                    loss_1 = F.l1_loss(restored_train, target_train) + F.l1_loss(fake_image_x8, target_train)+ F.l1_loss(fake_image_x4, target_train) + F.l1_loss(fake_image_x2, target_train)
                    loss_perpetual = loss_network(restored_train, target_train) + loss_network(fake_image_x8, target_train)+ loss_network(fake_image_x4, target_train) + loss_network(fake_image_x2, target_train)
                    gradient_loss = Gradient_loss(restored_train, target_train)+Gradient_loss(fake_image_x8, target_train)+Gradient_loss(fake_image_x4, target_train)+Gradient_loss(fake_image_x2, target_train)
                    loss = loss_1 + args.lp_weight * loss_perpetual + args.lg_weight * gradient_loss
                    
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epochLoss += loss.item()
            iters.set_description('Train !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, args.epoch, loss.item()))
        # validation
        if epoch % args.val_frequency == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_val, target_val = (x.cuda(), y.cuda()) if args.cuda else (x, y)
                with torch.no_grad():
                    if args.only_last:
                        restored_val = model_restoration(input_val, only_last=args.only_last)
                    else:
                        restored_val,_, _, _ = model_restoration(input_val, only_last=args.only_last)
                for restored_val, target_val in zip(restored_val.clamp_(-1, 1), target_val):
                    psnr_val_rgb.append(torchPSNR(restored_val, target_val))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                torch.save(model_restoration.state_dict(), args.save_state)
            print("----------------------------------------------------------------------------------------------")
            print("Validation Finished, Current PSNR: {:.4f}, Best PSNR: {:.4f}.".format(psnr_val_rgb, best_psnr))
            print("----------------------------------------------------------------------------------------------")
        # Save the last model
        torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(),
                'PSNR': best_psnr,
                }, args.resume_state)
        scheduler.step()
    print("Training Process Finished ! Best PSNR : {:.4f}".format(best_psnr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_val', type=int, default=1)
    parser.add_argument('--patch_size_train', type=int, default=256)
    parser.add_argument('--patch_size_val', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_min', type=float, default=1e-8)
    parser.add_argument('--train_data', type=str, default='D:\dataset_GRS\Haze1k_thin\\train')
    parser.add_argument('--val_data', type=str, default='D:\dataset_GRS\Haze1k_thin\\test')
    parser.add_argument('--resume_state', type=str, default='./thin_DF/model_latest.pth')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--save_state', type=str, default='./thin_DF/model_best.pth')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--val_frequency', type=int, default=1)
    parser.add_argument('--lp_weight', type=float, default=0.05)
    parser.add_argument('--lg_weight', type=float, default=0.08)
    parser.add_argument('--only_last', type=bool, default=False)
    parser.add_argument('--autocast', type=bool, default=True)
    parser.add_argument('--num_works', type=int, default=1)
    args = parser.parse_args()

    train(args)




