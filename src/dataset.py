import torch
from datasets import load_dataset
from torchvision import transforms


def get_dataset(config):
    dataset = load_dataset(config.dataset_name, split="train", cache_dir="data")

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
    return train_dataloader
