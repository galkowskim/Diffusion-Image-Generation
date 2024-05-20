import argparse
import os

import numpy as np
import torch
import yaml
from datasets import load_dataset
from schedulers import LinearNoiseScheduler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from unet import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config["diffusion_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((model_config["im_size"], model_config["im_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset = load_dataset("pcuenq/lsun-bedrooms", split="train", cache_dir="../data")
    dataset.set_transform(transform)
    data_loader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=4
    )

    model = Unet(model_config).to(device)
    model.train()

    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config["task_name"])

    if os.path.exists(
        os.path.join(train_config["task_name"], train_config["ckpt_name"])
    ):
        print("Loading checkpoint as found one")
        model.load_state_dict(
            torch.load(
                os.path.join(train_config["task_name"], train_config["ckpt_name"]),
                map_location=device,
            )
        )
    num_epochs = train_config["num_epochs"]
    optimizer = Adam(model.parameters(), lr=train_config["lr"])
    criterion = torch.nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(data_loader):
            optimizer.zero_grad()
            im = im["images"].float().to(device)

            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config["num_timesteps"], (im.shape[0],)).to(
                device
            )

            # add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Finished epoch:{epoch_idx + 1} | Loss : {np.mean(losses):.4f}")
        torch.save(
            model.state_dict(),
            os.path.join(train_config["task_name"], train_config["ckpt_name"]),
        )

    with open(os.path.join(train_config["task_name"], "config.yaml"), "w") as file:
        yaml.dump(config, file)

    print("Done Training ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument(
        "--config", dest="config_path", default="default.yaml", type=str
    )
    args = parser.parse_args()
    train(args)
