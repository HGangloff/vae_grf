import os
import sys
sys.path.insert(0, "../") 
import time
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
from utils import (get_train_dataloader,
                   get_test_dataloader,
                   load_model_parameters,
                   load_vqvae,
                   parse_args
                   )

def ssim(a, b, win_size):
    "Structural di-SIMilarity: SSIM"
    a = a.detach().cpu().permute(1, 2, 0).numpy()
    b = b.detach().cpu().permute(1, 2, 0).numpy()

    #b = gaussian_filter(b, sigma=2)

    try:
        score, full = structural_similarity(a, b, #multichannel=True,
            channel_axis=2, full=True, win_size=win_size, data_range=1)
    except ValueError: # different version of scikit img
        score, full = structural_similarity(a, b, multichannel=True,
            channel_axis=2, full=True, win_size=win_size, data_range=1)
    #return 1 - score, np.median(1 - full, axis=2)  # Return disim = (1 - sim)
    return 1 - score, np.prod((1 - full), axis=2)

def get_error_pixel_wise(model, x, loss="rec_loss"):
    x_rec, _ = model(x)
    
    return x_rec

def test(args):
    ''' livestock testing pipeline '''
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"

    predictions_dir ="./" + args.dataset + "_predictions"
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    # Load dataset
    train_dataloader, train_dataset = get_train_dataloader(
        args,
        fake_dataset_size=256,
    )
    # NOTE force test batch size to be 1
    args.batch_size_test = 1
    # fake_dataset_size=None leads a test on all the test dataset
    test_dataloader, test_dataset = get_test_dataloader(
        args,
        fake_dataset_size=None,
        with_loc=True
    )

    # Load model
    model = load_vqvae(args)
    model.to(device)

    try:
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir,
            checkpoints_saved_dir, device)
    except FileNotFoundError:
        raise RuntimeError("The model checkpoint does not exist !")

    dissimilarity_func = ssim

    classes = {}

    model.eval()

    aucs = []

    print(test_dataloader)
    pbar = tqdm(test_dataloader)
    for imgs, gt in pbar:
        imgs = imgs.to(device)
        if args.dataset in ["mvtec", "livestock"]:
            # gt is a segmentation mask
            gt_np = gt[0].permute(1, 2, 0).cpu().numpy()[..., 0]
            gt_np = (gt_np - np.amin(gt_np)) / (np.amax(gt_np) - np.amin(gt_np))

        with torch.no_grad():
            x_rec = get_error_pixel_wise(model, imgs)
            x_rec = model.mean_from_lambda(x_rec)

        if args.dataset == "mvtec":
            score, ssim_map = dissimilarity_func(x_rec[0], imgs[0], 15)
        if args.dataset == "livestock":
            score, ssim_map = dissimilarity_func(x_rec[0], imgs[0], 11)

        ssim_map = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map)
        - np.amin(ssim_map)))

        x_rec, _ = model(imgs)
        x_rec = model.mean_from_lambda(x_rec)

        mad = torch.mean(torch.abs(model.mu - torch.mean(model.mu,
            dim=(0,1))), dim=(0,1))

        mad = mad.detach().cpu().numpy()

        mad = ((mad - np.amin(mad)) / (np.amax(mad)
            - np.amin(mad)))

        mad = mad.repeat(8, axis=0).repeat(8, axis=1)

        # MAD metric
        amaps = mad

        # SM metric
        #amaps = ssim_map

        # MAD*SM metric
        amaps = mad * ssim_map

        amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps)
            - np.amin(amaps)))

        if args.dataset in ["mvtec", "livestock"]:
            preds = amaps.copy() 
            mask = np.zeros(gt_np.shape)

            try:
                auc = roc_auc_score(gt_np.astype(np.int8).flatten(), preds.flatten())
                aucs.append(auc)
            except ValueError:
                pass
                # ROCAUC will not be defined when one class only in y_true

        m_aucs = np.mean(aucs)
        pbar.set_description(f"mean ROCAUC: {m_aucs:.3f}")

        ori = imgs[0].permute(1, 2, 0).cpu().numpy()
        gt = gt[0].permute(1, 2, 0).cpu().numpy()
        rec = x_rec[0].detach().permute(1, 2, 0).cpu().numpy()
        path_to_save = args.dataset + '_predictions/'
        img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'ori.png')
        img_to_save = Image.fromarray((gt_np * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'gt.png')
        img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'rec.png')
        cm = plt.get_cmap('jet')
        amaps = cm(amaps)
        img_to_save = Image.fromarray((amaps[..., :3] * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'final_amap.png')

    m_auc = np.mean(aucs)
    print("Mean auc on", args.category, args.defect, m_auc)

    return m_auc

if __name__ == "__main__":
    args = parse_args()

    if args.dataset in ["mvtec", "livestock"]:
        m_auc = test(
            args,
            )
