import os
import logging
import wandb
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan
from experiments.cityscapes.model_curve import SegNetMtan_Curve
from experiments.cityscapes.utils import ConfMatrix, delta_fn, depth_error
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

    return loss


def main(path, lr, bs, device):
    # ----
    # Nets
    # ---
    # model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    # model = model.to(device)

    curve = NURBS # Bezier NURBS

    model = CurveNet(
        curve,
        SegNetMtan_Curve,
        args.num_bends,
        args.fix_start,
        args.fix_end
    )

    base_model = None
    for path_cur, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
        if path_cur is not None:
            if base_model is None:
                base_model = SegNetMtan()
            checkpoint = torch.load(path_cur)
            print('Loading %s as point #%d' % (path_cur, k))
            base_model.load_state_dict(checkpoint)
            model.import_base_parameters(base_model, k)

    # model.init_linear()
    model.cuda()
    
    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    cityscapes_test_set = Cityscapes(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=2, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    # optimizer = torch.optim.Adam(
    #     [
    #         dict(params=model.parameters(), lr=lr),
    #         dict(params=weight_method.parameters(), lr=args.method_params_lr),
    #     ],
    # )
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr = lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    regularizer = l2_regularizer(args.wd)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(7)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)
        alpha = 1.0 * epoch / epochs
        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth = train_depth.to(device)

            train_pred, _, t1 = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )
        
            train_pred_asst, _, t2 = model(train_data, return_representation=True)

            losses_asst = torch.stack(
                (
                    calc_loss(train_pred_asst[0], train_label, "semantic"),
                    calc_loss(train_pred_asst[1], train_depth, "depth"),
                )
            )

            loss_sum = (losses[0] + losses_asst[0]) / 2 + 10 * (losses[1] + losses_asst[1]) / 2

            pareto_weight = (torch.exp(-torch.abs(t1-0.5)) + torch.exp(-torch.abs(t2 - 0.5))) / 2.0
            loss_order = (torch.exp((losses[0] - losses_asst[0]) * (t2 - t1)) + torch.exp((losses[1] - losses_asst[1]) * (t1 - t2))) / 2.0
            loss = loss_sum + pareto_weight * loss_order

            loss.backward()

            loss_list.append(losses.detach().cpu())
            optimizer.step()

            if "famo" in args.method:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                        (
                            calc_loss(train_pred[0], train_label, "semantic"),
                            calc_loss(train_pred[1], train_depth, "depth"),
                        )
                    )
                    weight_method.method.update(new_losses.detach())

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(7)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth = test_depth.to(device)

                test_pred, _, _ = model(test_data, return_representation=True)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [7, 8, 10, 11]]
            )
            deltas[epoch] = test_delta_m

            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} "
                f" TEST: {avg_cost[epoch, 6]:.4f} {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} | "
                f"{avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}"
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)



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

            if "famo" in args.method:
                name = f"{args.method}_gamma{args.gamma}_sd{args.seed}"
            elif "fairgrad" in args.method:
                name = f"{args.method}_alpha{args.alpha}_sd{args.seed}"
            else:
                name = f"{args.method}_sd{args.seed}"

            torch.save({
                "delta_m": deltas,
                "keys": keys,
                "avg_cost": avg_cost,
                "losses": loss_list,
            }, f"./save/{name}.stats")
    
    model_name = args.ckpt 
    torch.save(model.state_dict(), model_name)



if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=8,
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
    # parser.add_argument('--fix_start', dest='fix_start', action='store_true',
    #                 help='fix start point (default: off)')
    parser.add_argument('--fix_start', default=True,
                    help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default='', metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
    # parser.add_argument('--fix_end', dest='fix_end', action='store_true',
    #                 help='fix end point (default: off)')
    parser.add_argument('--fix_end', default=True,
                    help='fix end point (default: off)')
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
