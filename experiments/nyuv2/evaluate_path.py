import os
import logging
import wandb
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.model_curve import SegNetMtan_Curve
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods

from curves import CurveNet, Bezier, l2_regularizer, NURBS
from utils import update_bn

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def main(path, lr, bs, device):
    # ----
    # Nets
    # ---
    # model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    # model = model.to(device)

    curve = NURBS # NURBS

    model = CurveNet(
        curve,
        SegNetMtan_Curve,
        args.num_bends,
    )

    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint)
    model.cuda()

    # base_model = None
    # for path_cur, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
    #     if path_cur is not None:
    #         if base_model is None:
    #             base_model = SegNetMtan()
    #         checkpoint = torch.load(path_cur)
    #         print('Loading %s as point #%d' % (path_cur, k))
    #         base_model.load_state_dict(checkpoint)
    #         model.import_base_parameters(base_model, k)
    
    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method]
    )

    regularizer = l2_regularizer(args.wd)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(13)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)[::-1][1:60]
    t = torch.FloatTensor([0.0]).cuda()
    for i, t_value in enumerate(ts):
        cost = np.zeros(24, dtype=np.float32)

        t.data.fill_(t_value)
        # weights = model.weights(t)
        # previous_weights = weights.copy()
        update_bn(train_loader, model, device, t=t)
        
        model.eval()

        # evaluate on the training set
        for j, batch in enumerate(train_loader):
            custom_step += 1

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, _, _ = model(train_data, True, t)

            losses = torch.stack(
                    (
                        calc_loss(train_pred[0], train_label, "semantic"),
                        calc_loss(train_pred[1], train_depth, "depth"),
                        calc_loss(train_pred[2], train_normal, "normal"),
                    )
            )

            loss_list.append(losses.detach().cpu())

            if "famo" in args.method:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                    )
                    weight_method.method.update(new_losses.detach())

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[i, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{i+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )

        # compute mIoU and acc
        avg_cost[i, 1:3] = conf_mat.get_metrics()

        conf_mat = ConfMatrix(13)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _, _ = model(test_data, True, t)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[i, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[i, 13:15] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[i, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )
            deltas[i] = test_delta_m


            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
            )
            print(
                f"Epoch: {i:04d} | TRAIN: {avg_cost[i, 0]:.4f} {avg_cost[i, 1]:.4f} {avg_cost[i, 2]:.4f} "
                f"| {avg_cost[i, 3]:.4f} {avg_cost[i, 4]:.4f} {avg_cost[i, 5]:.4f} | {avg_cost[i, 6]:.4f} "
                f"{avg_cost[i, 7]:.4f} {avg_cost[i, 8]:.4f} {avg_cost[i, 9]:.4f} {avg_cost[i, 10]:.4f} {avg_cost[i, 11]:.4f} || "
                f"TEST: {avg_cost[i, 12]:.4f} {avg_cost[i, 13]:.4f} {avg_cost[i, 14]:.4f} | "
                f"{avg_cost[i, 15]:.4f} {avg_cost[i, 16]:.4f} {avg_cost[i, 17]:.4f} | {avg_cost[i, 18]:.4f} "
                f"{avg_cost[i, 19]:.4f} {avg_cost[i, 20]:.4f} {avg_cost[i, 21]:.4f} {avg_cost[i, 22]:.4f} {avg_cost[i, 23]:.4f} "
                f"| {test_delta_m:.3f}"
            )


            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",

                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
            ]




if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200, 
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    parser.add_argument('--num_bends', type=int, default=4, metavar='N',
                    help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, default='', metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default='', metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', default=True,
                    help='fix end point (default: off)')
    parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
    parser.add_argument('--ckpt', type=str, default='', metavar='CKPT',
                    help='checkpoint to eval (default: None)')
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()
