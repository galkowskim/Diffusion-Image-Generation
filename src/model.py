import torch
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup


def get_model(config, num_training_steps):
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    if config.model_type == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif config.model_type == "ddim":
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    elif config.model_type == "pndm":
        noise_scheduler = PNDMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return model, noise_scheduler, optimizer, lr_scheduler
