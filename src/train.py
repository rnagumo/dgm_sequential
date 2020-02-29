
"""Training method"""

import argparse
import datetime
import pathlib

import torch
import tensorboardX as tb

import dgm_sequential.dataset as dsd
import dgm_sequential.model as dsm
import dgm_sequential.utils as dsu


def train(args, logger, config):

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # CUDA setting
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_num}" if use_cuda else "cpu")
    logger.info(f"Device: {device}")

    # Random seed
    torch.manual_seed(args.seed)

    # Tensorboard writer
    writer = tb.SummaryWriter(args.logdir)

    # Timer
    timer = datetime.datetime.now()

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------

    logger.info("Prepare data")

    # Loader
    batch_size = args.batch_size
    path = pathlib.Path(args.root, args.filename)
    train_loader, valid_loader, test_loader = dsd.init_poly_dataloader(
        path, use_cuda, batch_size)

    # Data dimension (seq_len, batch_size, input_size)
    x_dim = train_loader.dataset.data.size(2)
    t_dim = train_loader.dataset.data.size(0)

    # Sample data for reconstruction of size (batch_size, seq_len, input_size)
    x_org, _ = iter(test_loader).next()
    x_org = x_org[:8]

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
        model = dsm.DMM(**config["dmm_params"], **params)
    elif args.model == "srnn":
        model = dsm.SRNN(**config["srnn_params"], **params)
    elif args.model == "storn":
        model = dsm.STORN(**config["storn_params"], **params)
    elif args.model == "vrnn":
        model = dsm.VRNN(**config["vrnn_params"], **params)
    elif args.model == "tdvae":
        model = dsm.TDVAE(**config["tdvae_params"], **params)
    else:
        raise KeyError(f"Not implemented model is specified, {args.model}")

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = model.run(train_loader, epoch, training=True)
        valid_loss = model.run(valid_loader, epoch, training=False)
        test_loss = model.run(test_loader, epoch, training=False)

        logger.info(f"Train loss = {train_loss['loss']}")
        logger.info(f"Valid loss = {valid_loss['loss']}")
        logger.info(f"Test loss = {test_loss['loss']}")

        # Log
        for mode, dicts in zip(["train", "valid", "test"],
                               [train_loss, valid_loss, test_loss]):
            for key, value in dicts.items():
                writer.add_scalar(f"{mode}/{key}", value, epoch)

        # Sample data
        if epoch % args.plot_interval == 0:
            logger.info("Sample data")
            x_sample, z_sample = model.sample()
            writer.add_images("sample/latent", z_sample, epoch)
            writer.add_images("sample/observable", x_sample, epoch)

            x_sample, z_sample = model.reconstruct(x_org, time_step=10)
            writer.add_images("reconstruct/original", x_org[:, None], epoch)
            writer.add_images("reconstruct/observable", x_sample, epoch)
            writer.add_images("reconstruct/latent", z_sample, epoch)

        # Save model
        if epoch % args.save_interval == 0:
            logger.info(f"Save model at epoch {epoch}")
            t = timer.strftime("%Y%m%d%H%M%S")
            filename = f"{args.model}_{t}_epoch_{epoch}.pt"
            torch.save({"distributions_dict": model.distributions.state_dict(),
                        "optimizer_dict": model.optimizer.state_dict()},
                       pathlib.Path(args.logdir, filename))

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
    parser.add_argument("--logdir", type=str, default="../logs/tmp/")
    parser.add_argument("--root", type=str, default="../data/poly/")
    parser.add_argument("--filename", type=str, default="JSB_Chorales.pickle")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model", type=str, default="dmm")
    parser.add_argument("--cuda-num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--plot-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)

    return parser.parse_args()


def main():
    # Args
    args = init_args()

    # Make logdir
    dsu.check_logdir(args.logdir)

    # Logger
    logger = dsu.init_logger(args.logdir)
    logger.info("Start logger")
    logger.info(f"Commant line args: {args}")

    # Config
    config = dsu.load_config(args.config)
    logger.info(f"Configs: {config}")

    try:
        train(args, logger, config)
    except Exception as e:
        logger.exception(f"Run function error: {e}")

    logger.info("End logger")


if __name__ == "__main__":
    main()
