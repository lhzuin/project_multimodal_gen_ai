# train.py
import os, sys, signal, random
import numpy as np
import torch
import wandb
import hydra

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from transformers.optimization import get_scheduler
from tqdm.auto import tqdm

os.environ["HYDRA_FULL_ERROR"] = "1"

OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler_for_stage(optimizer, *, stage_dict, cfg, num_training_steps):
    scheduler_type = stage_dict["lr_scheduler"]
    warmup_fraction = stage_dict["warmup_fraction"]
    use_warmup = getattr(cfg, "use_warmup", True)
    num_warmup_steps = int(warmup_fraction * num_training_steps) if use_warmup else 0

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def param_groups_for(model, cfg, stage_dict):
    groups = model.get_param_groups()
    stage_groups = stage_dict.get("groups", {})
    optim_cfg = cfg.optim

    def get_lr(key, fallback="lr"):
        if hasattr(optim_cfg, key):
            return getattr(optim_cfg, key)
        if hasattr(optim_cfg, fallback):
            return getattr(optim_cfg, fallback)
        raise ValueError(f"Could not find LR '{key}' or '{fallback}' in cfg.optim")

    pg = []
    for group_name, lr_key in stage_groups.items():
        if group_name not in groups:
            continue
        params = [p for p in groups[group_name] if p.requires_grad]
        if not params:
            continue
        lr = get_lr(lr_key)
        weight_decay = cfg.optim.weight_decay
        pg.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    return pg


def split_indices(n: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_fraction * n))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def batch_is_empty(batch: dict) -> bool:
    # your collate returns {"images": torch.empty(0)} when nothing valid
    return (not isinstance(batch, dict)) or ("images" not in batch) or (batch["images"].numel() == 0)


class Trainer:
    def __init__(self, cfg, logger, device):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.model = None
        self.loss_fn = None
        self.scaler = GradScaler(device="cuda") if device.type == "cuda" else None

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.optimizer = None

        self.patience = cfg.early_stopping.patience
        self.min_epochs = cfg.early_stopping.min_epochs
        self.checkpoint_save_path = cfg.checkpoint_save_path
        self.acc_epsilon = cfg.acc_epsilon

        # Build opt cfg like before (keep only AdamW-safe args)
        opt_cfg_full = OmegaConf.to_container(cfg.optim, resolve=True, enum_to_str=True)
        target = opt_cfg_full.pop("_target_")
        allowed_keys = {
            "lr", "betas", "eps", "weight_decay",
            "amsgrad", "foreach", "maximize",
            "capturable", "differentiable", "fused",
        }
        opt_kwargs = {k: v for k, v in opt_cfg_full.items() if k in allowed_keys}
        opt_kwargs["_target_"] = target
        self.opt_cfg = opt_kwargs

    def init_data(self):
        # Dataset + collate from Hydra
        self.train_dataset = hydra.utils.instantiate(self.cfg.dataset_train)
        self.val_dataset   = hydra.utils.instantiate(self.cfg.dataset_val)
        collate_fn = hydra.utils.instantiate(self.cfg.collate)


        # Split into train/val
        val_fraction = float(getattr(self.cfg.data, "val_fraction", 0.05))
        train_idx, val_idx = split_indices(len(self.train_dataset), val_fraction, seed=int(self.cfg.seed))

        train_ds = Subset(self.train_dataset, train_idx)
        val_ds   = Subset(self.val_dataset, val_idx)


        dl_cfg = self.cfg.dataloader
        self.train_loader = DataLoader(
            train_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=dl_cfg.shuffle,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=getattr(dl_cfg, "drop_last", False),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=False,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=False,
        )

        print(f"Dataset size={len(self.train_dataset)} | train={len(train_ds)} | val={len(val_ds)}")

    def init_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model.instance).to(self.device)
        self.loss_fn = hydra.utils.instantiate(self.cfg.loss_fn)

    def _forward_loss_and_metrics(self, batch):
        # batch["images"]: [B,3,H,W]
        # batch["labels64"]: [B,64]
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels64"].to(self.device, non_blocking=True)  # [B,64]

        logits = self.model(images)  # expected [B,64,13]
        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B,64,13], got shape={tuple(logits.shape)}")

        B, S, C = logits.shape
        if S != 64:
            raise ValueError(f"Expected 64 square tokens, got S={S}")

        loss = self.loss_fn(logits.reshape(B * S, C), labels.reshape(B * S))

        preds = logits.argmax(dim=-1)  # [B,64]
        correct = (preds == labels).sum().item()
        total = labels.numel()

        return loss, correct, total

    def run_epoch(self, stage_name, scheduler):
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        progress = tqdm(self.train_loader, desc=f"{stage_name} | training", leave=False)
        for batch in progress:
            if batch_is_empty(batch):
                continue

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with autocast(device_type=self.device.type):
                    loss, correct, n = self._forward_loss_and_metrics(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, correct, n = self._forward_loss_and_metrics(batch)
                loss.backward()
                self.optimizer.step()

            if scheduler:
                scheduler.step()

            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val * n
            total_correct += correct
            total_n += n

            acc_val = correct / max(1, n)
            progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.3f}")

            if self.logger:
                wandb.log({
                    f"{stage_name}/loss_step": loss_val,
                    f"{stage_name}/acc_step": acc_val,
                    "aggregate/loss_step": loss_val,
                })

        return (total_loss / max(1, total_n)), (total_correct / max(1, total_n))

    @torch.no_grad()
    def run_validation(self):
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in self.val_loader:
            if batch_is_empty(batch):
                continue

            loss, correct, n = self._forward_loss_and_metrics(batch)
            total_loss += float(loss.detach().cpu().item()) * n
            total_correct += correct
            total_n += n

        val_loss = total_loss / max(1, total_n)
        val_acc = total_correct / max(1, total_n)
        return val_loss, val_acc

    def run_stage(self, stage_dict):
        stage_name = stage_dict["name"]
        print(f"\nRunning stage: {stage_name}")
        print(">>> Optim cfg:", self.cfg.optim)

        # Freeze/unfreeze as before
        self.model.set_trainable_groups(list(stage_dict["groups"].keys()))

        self.optimizer = hydra.utils.instantiate(
            self.opt_cfg,
            params=param_groups_for(self.model, self.cfg, stage_dict),
            _convert_="all",
        )

        scheduler = None
        if self.cfg.use_warmup:
            num_training_steps = stage_dict["epochs"] * len(self.train_loader)
            scheduler = build_scheduler_for_stage(
                self.optimizer,
                stage_dict=stage_dict,
                cfg=self.cfg,
                num_training_steps=num_training_steps,
            )

        best_val_loss, best_val_acc, epochs_no_improve = float("inf"), 0.0, 0

        for epoch in range(stage_dict["epochs"]):
            train_loss, train_acc = self.run_epoch(stage_name, scheduler)
            val_loss, val_acc = self.run_validation()

            print(
                f"[{stage_name}][Epoch {epoch}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if self.logger:
                wandb.log({
                    "epoch": epoch,
                    f"{stage_name}/loss_epoch": train_loss,
                    f"{stage_name}/acc_epoch": train_acc,
                    f"{stage_name}/val_loss_epoch": val_loss,
                    f"{stage_name}/val_acc_epoch": val_acc,
                    "aggregate/val_loss_epoch": val_loss,
                    "aggregate/val_acc_epoch": val_acc,
                })

            improved = (val_acc > best_val_acc) or (
                (best_val_acc - val_acc < self.acc_epsilon) and (val_loss < best_val_loss)
            )

            if improved:
                best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
                torch.save(self.model.state_dict(), self.checkpoint_save_path)
                print(f"  -> saved best: val_acc={best_val_acc:.4f} val_loss={best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience and epoch >= self.min_epochs:
                    print(f"Early stopping {stage_name}.")
                    break

    def run_train(self):
        for i, stage_dict in enumerate(self.cfg.train_stages):
            if i > 0:
                print("Reloading best model from previous stage.")
                state = torch.load(self.checkpoint_save_path, map_location=self.device)
                self.model.load_state_dict(state)
            self.run_stage(stage_dict)


@hydra.main(config_path="../configs", config_name="chess_classifier_vit_v2", version_base="1.1")
def main(cfg):
    set_seed(int(cfg.seed))

    logger = wandb.init(project=cfg.wandb_project, name=cfg.experiment_name) if cfg.log else None

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device={device} | PID={os.getpid()}  (kill -SIGUSR1 {os.getpid()} to checkpoint+exit)")

    trainer = Trainer(cfg=cfg, logger=logger, device=device)
    trainer.init_data()
    trainer.init_model()

    def save_and_exit(*_):
        torch.save(trainer.model.state_dict(), cfg.checkpoint_save_path)
        print(f"Saved checkpoint → {cfg.checkpoint_save_path}")
        sys.exit(0)

    signal.signal(signal.SIGUSR1, save_and_exit)

    trainer.run_train()

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()