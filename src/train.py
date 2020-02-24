
"""Training method"""

import argparse
import pathlib

import torch
from torch.utils import tensorboard

from dataset.polydata import init_poly_dataloader
from model.dmm import DMM
from model.srnn import SRNN
from model.storn import STORN
from model.vrnn import VRNN
from utils.utils import init_logger, load_config, check_logdir


def train(args, logger, config):

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # CUDA setting
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_num}" if use_cuda else "cpu")
    logger.info(f"Device: {device}")

    # Random seed
    torch.manual_seed(args.seed)

    # Tensorboard writer
    writer = tensorboard.SummaryWriter(args.logdir)

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------

    # Loader
    batch_size = args.batch_size
    path = pathlib.Path(args.root, args.filename)
    train_loader, valid_loader, test_loader = init_poly_dataloader(
        path, use_cuda, batch_size)

    # Data dimension (seq_len, batch_size, input_size)
    x_dim = train_loader.dataset.data.size(2)
    t_dim = train_loader.dataset.data.size(0)

    # Log
    logger.info(f"Train data size: {train_loader.dataset.data.size()}")
    logger.info(f"Valid data size: {valid_loader.dataset.data.size()}")
    logger.info(f"Test data size: {test_loader.dataset.data.size()}")

    # -------------------------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------------------------

    params = {"x_dim": x_dim, "t_dim": t_dim, "device": device,
              "anneal_params": config["anneal_params"],
              "optimizer_params": config["optimizer_params"]}

    if args.model == "dmm":
        model = DMM(**config["dmm_params"], **params)
    elif args.model == "srnn":
        model = SRNN(**config["srnn_params"], **params)
    elif args.model == "storn":
        model = STORN(**config["storn_params"], **params)
    elif args.model == "vrnn":
        model = VRNN(**config["vrnn_params"], **params)
    else:
        raise KeyError

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = model.run(train_loader, epoch, training=True)
        valid_loss = model.run(valid_loader, epoch, training=False)
        test_loss = model.run(test_loader, epoch, training=False)

        # Log
        writer.add_scalar("loss/train_loss", train_loss["loss"], epoch)
        writer.add_scalar("loss/valid_loss", valid_loss["loss"], epoch)
        writer.add_scalar("loss/test_loss", test_loss["loss"], epoch)
        writer.add_scalar("training/annealing_factor",
                          train_loss["annealing_factor"], epoch)
        writer.add_scalar("training/cross_entropy",
                          train_loss["cross_entropy"], epoch)
        writer.add_scalar("training/kl_divergence",
                          train_loss["kl_divergence"], epoch)

        # Sample data
        if epoch % args.plot_interval == 0:
            sample = model.sample()
            writer.add_images("image_from_latent", sample, epoch)

        logger.info(f"Train loss = {train_loss['loss']}")
        logger.info(f"Valid loss = {valid_loss['loss']}")
        logger.info(f"Test loss = {test_loss['loss']}")

    # Log hyper-parameters
    hparam_dict = vars(args)
    hparam_dict.update(config["anneal_params"])
    hparam_dict.update(config["optimizer_params"])
    hparam_dict.update(config[f"{args.model}_params"])
    metric_dict = {"summary/train_loss": train_loss["loss"],
                   "summary/valid_loss": valid_loss["loss"],
                   "summary/test_loss": test_loss["loss"]}
    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()


def init_args():
    parser = argparse.ArgumentParser(description="Polyphonic data training")

    # Direcotry settings
    parser.add_argument("--logdir", type=str, default="../logs/seq/tmp/")
    parser.add_argument("--root", type=str, default="../data/poly/")
    parser.add_argument("--filename", type=str, default="JSB_Chorales.pickle")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model", type=str, default="dmm")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--plot-interval", type=int, default=100)

    return parser.parse_args()


def main():
    # Args
    args = init_args()

    # Make logdir
    check_logdir(args.logdir)

    # Logger
    logger = init_logger(args.logdir)
    logger.info("Start logger")
    logger.info(f"Commant line args: {args}")

    # Config
    config = load_config(args.config)
    logger.info(f"Configs: {config}")

    try:
        train(args, logger, config)
    except Exception as e:
        logger.exception(f"Run function error: {e}")

    logger.info("End logger")


if __name__ == "__main__":
    main()