import argparse
import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDIMPipeline, DDPMPipeline, PNDMPipeline
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from tqdm.auto import tqdm

from config import TrainingConfig
from dataset import get_dataset
from model import get_model


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size, generator=torch.manual_seed(config.seed)
    ).images
    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(
    config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler
):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    with open(f"{config.output_dir}/config.pkl", "wb") as f:
        pickle.dump(config, f)

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for _, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if config.model_type == "ddpm":
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
                )
            elif config.model_type == "ddim":
                pipeline = DDIMPipeline(
                    unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
                )
            elif config.model_type == "pndm":
                pipeline = PNDMPipeline(
                    unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
                )
            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, default="ddpm")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, required=True, default=10)
    args = parser.parse_args()
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        model_type=args.model_type,
    )
    train_dataloader = get_dataset(config)
    num_training_steps = len(train_dataloader) * config.num_epochs
    model, noise_scheduler, optimizer, lr_scheduler = get_model(
        config, num_training_steps
    )
    train_loop(
        config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler
    )
