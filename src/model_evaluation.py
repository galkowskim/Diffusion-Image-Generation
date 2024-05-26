import argparse
import os
import pickle

import numpy as np
import torch
from datasets import load_dataset
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    PNDMPipeline,
    PNDMScheduler,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm.auto import tqdm

from config import TrainingConfig


def main(experiment_dir, no_times, size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "pcuenq/lsun-bedrooms"
    dataset = load_dataset(dataset_name, split="train", cache_dir="data")

    with open(os.path.join(experiment_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
        assert isinstance(config, TrainingConfig)

    if config.model_type == "ddpm":
        pipeline = DDPMPipeline.from_pretrained(experiment_dir).to(device)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif config.model_type == "ddim":
        pipeline = DDIMPipeline.from_pretrained(experiment_dir).to(device)
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        noise_scheduler.set_timesteps(1000, device)
    elif config.model_type == "pndm":
        pipeline = PNDMPipeline.from_pretrained(experiment_dir).to(device)
        noise_scheduler = PNDMScheduler(num_train_timesteps=1000)
        noise_scheduler.timesteps = torch.from_numpy(
            np.arange(0, 1000)[::-1].copy()
        )  # it was not implemented correctly in HF, so need to add it manually
        noise_scheduler.set_timesteps(1000, device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True
    )

    fid = FrechetInceptionDistance(normalize=True)
    model = pipeline.unet.to(device)

    for _ in tqdm(range(no_times)):
        selected_images = []

        for batch in train_dataloader:
            real_images = batch["images"]
            random_indices = torch.randperm(real_images.size(0))[:size]
            selected_images.append(real_images[random_indices])

            if len(selected_images) * real_images.size(0) >= size:
                break

        real_images = torch.cat(selected_images, dim=0)[:size]

        noise = torch.randn(real_images.shape)
        timesteps = torch.LongTensor([999])
        noisy_images = noise_scheduler.add_noise(real_images, noise, timesteps)

        sample = noisy_images.to(device)

        torch.cuda.empty_cache()
        for i, t in tqdm(
            enumerate(noise_scheduler.timesteps), total=len(noise_scheduler.timesteps)
        ):
            with torch.no_grad():
                residual = model(sample, t).sample
            sample = noise_scheduler.step(residual, t, sample).prev_sample

        fake_images = sample.cpu()

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

    fid_score = float(fid.compute())
    print(f"FID: {fid_score}")

    with open(os.path.join(experiment_dir, "fid.txt"), "w") as f:
        f.write(f"FID = {fid_score}\n")
        f.write(f"{no_times = }, {size = }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DDPM experiment and compute FID score."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Directory containing the experiment configuration and model.",
    )
    parser.add_argument(
        "--no_times",
        type=int,
        default=5,
        help="Number of times to run the noise addition and sampling process.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Number of images to process in each iteration.",
    )

    args = parser.parse_args()
    main(args.experiment_dir, args.no_times, args.size)
