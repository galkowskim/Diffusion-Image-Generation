from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    image_size: int = 128
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    mixed_precision: str = "fp16"
    output_dir: str = "hg-im-size-128-epochs-50-pndm"
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0
    dataset_name: str = "pcuenq/lsun-bedrooms"
    model_type: Literal["ddpm", "ddim", "pndm"] = "pndm"
