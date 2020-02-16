
"""Training method"""

import argparse
import pathlib

import tqdm

import torch
from torch.utils import tensorboard

from dataset.polydata import init_poly_dataloader
from model.dmm import load_dmm_model, init_dmm_var, get_dmm_sample
from model.srnn import load_srnn_model, init_srnn_var, get_srnn_sample
from model.storn import load_storn_model, init_storn_var, get_storn_sample
from model.vrnn import load_vrnn_model, init_vrnn_var, get_vrnn_sample
from utils.utils import init_logger, load_config, check_logdir


def data_loop(epoch, loader, model, args, config, train_mode=True):

    device = config["device"]

    # Returned values
    total_loss = 0
    total_len = 0

    # Train with mini-batch
    for x, seq_len in tqdm.tqdm(loader):
        # Input dimension must be (timestep_size, batch_size, feature_size)
        x = x.transpose(0, 1).to(device)
        data = {"x": x}
        minibatch_size = x.size(1)

        # Mask for sequencial data
        mask = torch.zeros(x.size(0), x.size(1)).to(device)
        for i, v in enumerate(seq_len):
            mask[:v, i] += 1

        # Initialize latent variable
        data.update(config["init_func"](minibatch_size, config, x=x))

        # Train / test
        if train_mode:
            _loss = model.train(data, mask=mask, epoch=epoch,
                                writer=config["writer"])
        else:
            _loss = model.test(data, mask=mask, epoch=epoch,
                               writer=config["writer"])

        # Add training results
        total_loss += _loss * minibatch_size
        total_len += seq_len.sum()

    return total_loss / total_len


def draw_image(sampler, args, config):

    # Get update parameters
    data = config["init_func"](1, config)

    x = []
    with torch.no_grad():
        for _ in range(config["t_dim"]):
            # Sample
            x_t, data = config["sample_func"](sampler, data)

            # Add to data list
            x.append(x_t)

        # Data of size (batch_size, seq_len, input_size)
        x = torch.cat(x).transpose(0, 1)

        # Return data of size (1, batch_size, seq_len, input_size)
        return x[:, None]


def train(args, logger, config):

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Settings
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    config["device"] = device

    # Tensorboard writer
    writer = tensorboard.SummaryWriter(args.logdir)
    config["writer"] = writer

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------

    # Loader
    batch_size = args.batch_size
    path = pathlib.Path(args.root, args.filename)
    train_loader, valid_loader, test_loader = init_poly_dataloader(
        path, use_cuda, batch_size)

    logger.info(f"Train data size: {train_loader.dataset.data.size()}")
    logger.info(f"Valid data size: {valid_loader.dataset.data.size()}")
    logger.info(f"Test data size: {test_loader.dataset.data.size()}")

    # Data dimension (seq_len, batch_size, input_size)
    x_dim = train_loader.dataset.data.size(2)
    t_dim = train_loader.dataset.data.size(0)
    config.update({"x_dim": x_dim, "t_dim": t_dim})

    # -------------------------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------------------------

    model, sampler = config["load_func"](config)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = data_loop(epoch, train_loader, model, args, config, True)
        valid_loss = data_loop(epoch, valid_loader, model, args, config, False)
        test_loss = data_loop(epoch, test_loader, model, args, config, False)

        # Sample data
        sample = draw_image(sampler, args, config)

        # Log
        writer.add_scalar("Loss/train_loss", train_loss.item(), epoch)
        writer.add_scalar("Loss/valid_loss", valid_loss.item(), epoch)
        writer.add_scalar("Loss/test_loss", test_loss.item(), epoch)
        writer.add_images("image_from_latent", sample, epoch)

        logger.info(f"Train loss = {train_loss.item()}")
        logger.info(f"Valid loss = {valid_loss.item()}")
        logger.info(f"Test loss = {test_loss.item()}")

    # Log hyper-parameters
    hparam_dict = vars(args)
    for key, d in config.items():
        if "_params" in key:
            hparam_dict.update(d)

    metric_dict = {"train_loss": train_loss.item(),
                   "valid_loss": valid_loss.item(),
                   "test_loss": test_loss.item()}
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
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)

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

    # Model
    load_func = {"dmm": load_dmm_model, "vrnn": load_vrnn_model,
                 "srnn": load_srnn_model, "storn": load_storn_model}
    init_func = {"dmm": init_dmm_var, "vrnn": init_vrnn_var,
                 "srnn": init_srnn_var, "storn": init_storn_var}
    sample_func = {"dmm": get_dmm_sample, "vrnn": get_vrnn_sample,
                   "srnn": get_srnn_sample, "storn": get_storn_sample}
    config.update({
        "load_func": load_func[args.model],
        "init_func": init_func[args.model],
        "sample_func": sample_func[args.model],
    })

    try:
        train(args, logger, config)
    except Exception as e:
        logger.exception(f"Run function error: {e}")

    logger.info("End logger")


if __name__ == "__main__":
    main()
