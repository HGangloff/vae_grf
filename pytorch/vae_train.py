import os
import sys
sys.path.insert(0, "../") 
import argparse
import numpy as np
import time

from torchvision import transforms, utils
import torch
from torch import nn

from utils import (get_train_dataloader,
                   get_test_dataloader,
                   load_model_parameters,
                   load_vqvae,
                   update_loss_dict,
                   print_loss_logs,
                   parse_args
                   )

def train(model, train_loader, device, optimizer, epoch, optimizer2=None):

    model.train()
    train_loss = 0
    loss_dict = {}

    for batch_idx, (input_mb, lbl) in enumerate(train_loader):
        print(batch_idx + 1, end=", ", flush=True)
        input_mb = input_mb.to(device)
        lbl = lbl.to(device)
        optimizer.zero_grad() # otherwise grads accumulate in backward
        if optimizer2 is not None:
            optimizer2.zero_grad()

        loss, recon_mb, loss_dict_new = model.step(
            input_mb
        )

        (-loss).backward()
        train_loss += loss.item()
        loss_dict = update_loss_dict(loss_dict, loss_dict_new)
        optimizer.step()
        if optimizer2 is not None:
            optimizer2.step()
            logrange_prior = dict(model.named_parameters())["logrange_prior"]
            logsigma_prior = dict(model.named_parameters())["logsigma_prior"]
            if torch.exp(logrange_prior) < 0.1:
                logrange_prior.data = np.log(0.1) * torch.ones_like(logrange_prior)
            if torch.exp(logrange_prior) > 10:
                logrange_prior.data = np.log(10.) * torch.ones_like(logrange_prior)
            if torch.exp(logsigma_prior)  ** 2< 0.001:
                logsigma_prior.data = np.log(np.sqrt(0.001)) * torch.ones_like(logsigma_prior)
            if torch.exp(logsigma_prior) > 5:
                logsigma_prior.data = np.log(5.) * torch.ones_like(logsigma_prior)
    nb_mb_it = len(train_loader.dataset) // input_mb.shape[0]
    train_loss /= nb_mb_it
    loss_dict = {k:v / nb_mb_it for k, v in loss_dict.items()}
    return train_loss, input_mb, recon_mb, loss_dict, lbl


def eval(model, test_loader, device):
    model.eval()
    input_mb, gt_mb = next(iter(test_loader))
    gt_mb = gt_mb.to(device)
    input_mb = input_mb.to(device)
    recon_mb, opt_out = model(input_mb)
    recon_mb = model.mean_from_lambda(recon_mb)
    return input_mb, recon_mb, gt_mb, opt_out


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Cuda available ?", torch.cuda.is_available())
    print("Pytorch device:", device)
    seed = 11
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = load_vqvae(args)
    model.to(device)

    train_dataloader, train_dataset = get_train_dataloader(args)
    test_dataloader, test_dataset = get_test_dataloader(args)

    nb_channels = args.nb_channels

    img_size = args.img_size
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test

    print("Nb channels", nb_channels, "img_size", img_size,
        "mini batch size", batch_size)


    out_dir = './torch_logs'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"
    res_dir = './torch_results'
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    data_dir = './torch_datasets'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)


    try:
        if args.force_train:
            raise FileNotFoundError
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir,
            checkpoints_saved_dir, device)
    except FileNotFoundError:
        print("Starting training")
        #print([p for p in model.parameters()])
        if args.model == "vae_grf":
            #parameter_names = ['logrange_prior']
            parameter_names = ['logsigma_prior', 'logrange_prior']
            base_params = [p[1] for p in filter(
                    lambda p: ((p[0] not in parameter_names) and
                        (p[1].requires_grad)),
                    model.named_parameters()
                    )]
            vae_params = [p[1] for p in filter(
                    lambda p: ((p[0] in parameter_names) and
                        (p[1].requires_grad)),
                    model.named_parameters()
                    )]
            optimizer = torch.optim.Adam(
                #[{'params':base_params},
                #{'params':vae_params,
                #    'lr':args.lr
                #    }
                #],
                base_params,
                lr=args.lr
            )
            optimizer2 = torch.optim.Rprop(
                vae_params,
                lr=1e-10
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr
            )
            optimizer2 = None
        for epoch in range(args.num_epochs):
            print("Epoch", epoch + 1)
            #if epoch < 10:
            #    loss, input_mb, recon_mb, loss_dict, lbl = train(
            #        model=model,
            #        train_loader=train_dataloader,
            #        device=device,
            #        optimizer=optimizer,
            #        epoch=epoch,
            #        optimizer2=None)
            #else:
            loss, input_mb, recon_mb, loss_dict, lbl = train(
                model=model,
                train_loader=train_dataloader,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
                optimizer2=optimizer2)
            print('epoch [{}/{}], train loss: {:.4f}'.format(
                epoch + 1, args.num_epochs, loss))
            try:
                print(f"VAEGRF estimated range={torch.exp(model.logrange_prior)}"
                      f" sigma={torch.exp(model.logsigma_prior) ** 2}")
            except AttributeError:
                pass

            # print loss logs
            f_name = os.path.join(out_dir, f"{args.exp}_loss_values.txt")
            print_loss_logs(f_name, out_dir, loss_dict, epoch, args.exp)
                    
            # save model parameters
            if (epoch + 1) % 100 == 0 or epoch in [0, 4, 9, 24, 49]:
                # to resume a training optimizer state dict and epoch
                # should also be saved
                torch.save(model.state_dict(), os.path.join(
                    checkpoints_dir, f"{args.exp}_{epoch + 1}.pth"
                    )
                )

            # print some reconstrutions
            if (epoch + 1) % 50 == 0 or epoch in [0, 4, 9, 14, 19, 24, 29, 49]:
                img_train = utils.make_grid(
                    torch.cat((
                        input_mb,
                        recon_mb,
                    ), dim=0), nrow=batch_size
                )
                utils.save_image(
                    img_train,
                    f"torch_results/{args.exp}_img_train_{epoch + 1}.png"
                )
                model.eval()
                input_test_mb, recon_test_mb, _, opt_out = eval(model=model,
                    test_loader=test_dataloader,
                    device=device)
                
                model.train()
                img_test = utils.make_grid(
                    torch.cat((
                        input_test_mb,
                        recon_test_mb),
                        dim=0),
                        nrow=batch_size_test
                )
                utils.save_image(
                    img_test,
                    f"torch_results/{args.exp}_img_test_{epoch + 1}.png"
                )

if __name__ == "__main__":
    args = parse_args()
    if args.category == "all":
        for cat in ["wood", "hazelnut", "pill", "leather", "carpet", "tile",
        "metal_nut", "capsule", "cable", "bottle", "toothbrush", "transistor",
        "zipper", "grid", "screw"]:
            args.category = cat
            main(args)
    if args.category == "all_restricted":
        for cat in ["wood", "hazelnut", "leather", "carpet", "tile", "grid"]:
            args.category = cat
            main(args)
    else:
        main(args)
