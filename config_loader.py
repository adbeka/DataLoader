"""Configuration loader for weed detection training."""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DataConfig:
    train_img_dir: str
    train_lbl_dir: str
    val_img_dir: str
    val_lbl_dir: str
    test_img_dir: str
    test_lbl_dir: str
    img_size: List[int]


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    num_classes: int
    lr: float
    lr_scheduler: str
    weight_decay: float


@dataclass
class ModelConfig:
    architecture: str
    backbone: str
    num_classes: int


@dataclass
class AugmentationConfig:
    horizontal_flip: float
    enable_advanced_aug: bool


@dataclass
class CheckpointingConfig:
    output_dir: str
    save_every_n_epochs: int
    keep_best_model: bool


@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    log_interval: int
    verbose: bool


@dataclass
class Config:
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    device: str

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load config from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict["data"]),
            training=TrainingConfig(**config_dict["training"]),
            model=ModelConfig(**config_dict["model"]),
            augmentation=AugmentationConfig(**config_dict["augmentation"]),
            checkpointing=CheckpointingConfig(**config_dict["checkpointing"]),
            logging=LoggingConfig(**config_dict["logging"]),
            device=config_dict["device"],
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "model": self.model.__dict__,
            "augmentation": self.augmentation.__dict__,
            "checkpointing": self.checkpointing.__dict__,
            "logging": self.logging.__dict__,
            "device": self.device,
        }
