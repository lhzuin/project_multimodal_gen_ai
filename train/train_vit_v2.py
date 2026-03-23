# train.py
import os, sys, signal, random, math, csv
import numpy as np
import torch
import wandb
import hydra

from copy import deepcopy
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
        self.val_dataset_noaug = None
        self.val_dataset_aug = None

        self.train_loader = None
        self.val_loader_noaug = None
        self.val_loader_aug = None

        self.optimizer = None

        self.patience = cfg.early_stopping.patience
        self.min_epochs = cfg.early_stopping.min_epochs
        self.checkpoint_save_path = cfg.checkpoint_save_path
        self.acc_epsilon = cfg.acc_epsilon

        self.history_path = getattr(cfg, "history_csv_path", "training_history.csv")
        self.history_rows = []

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

    def _build_loader(self, dataset, collate_fn, dl_cfg, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=dl_cfg.batch_size,
            shuffle=shuffle,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=drop_last,
        )

    def init_data(self):
        collate_fn = hydra.utils.instantiate(self.cfg.collate)

        # Train dataset
        self.train_dataset = hydra.utils.instantiate(self.cfg.dataset_train)

        # Build TWO validation datasets with same underlying samples but different augmentation flags
        val_cfg_noaug = OmegaConf.to_container(self.cfg.dataset_val, resolve=True)
        val_cfg_aug = deepcopy(val_cfg_noaug)
        val_cfg_noaug["apply_augmentations"] = False
        val_cfg_aug["apply_augmentations"] = True

        self.val_dataset_noaug = hydra.utils.instantiate(val_cfg_noaug)
        self.val_dataset_aug = hydra.utils.instantiate(val_cfg_aug)

        # Split indices ONCE, reuse same val subset for both noaug/aug validation
        val_fraction = float(getattr(self.cfg.data, "val_fraction", 0.05))
        train_idx, val_idx = split_indices(len(self.train_dataset), val_fraction, seed=int(self.cfg.seed))

        train_ds = Subset(self.train_dataset, train_idx)
        val_ds_noaug = Subset(self.val_dataset_noaug, val_idx)
        val_ds_aug = Subset(self.val_dataset_aug, val_idx)

        dl_cfg = self.cfg.dataloader
        self.train_loader = self._build_loader(
            train_ds, collate_fn, dl_cfg, shuffle=dl_cfg.shuffle, drop_last=getattr(dl_cfg, "drop_last", False)
        )
        self.val_loader_noaug = self._build_loader(
            val_ds_noaug, collate_fn, dl_cfg, shuffle=False, drop_last=False
        )
        self.val_loader_aug = self._build_loader(
            val_ds_aug, collate_fn, dl_cfg, shuffle=False, drop_last=False
        )

        print(
            f"Dataset size={len(self.train_dataset)} | "
            f"train={len(train_ds)} | val={len(val_ds_noaug)}"
        )

    def init_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model.instance).to(self.device)
        self.loss_fn = hydra.utils.instantiate(self.cfg.loss_fn)

    def _forward_loss_and_metrics(self, batch):
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels64"].to(self.device, non_blocking=True)

        logits = self.model(images)  # [B,64,13]
        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B,64,13], got shape={tuple(logits.shape)}")

        B, S, C = logits.shape
        if S != 64:
            raise ValueError(f"Expected 64 square tokens, got S={S}")

        loss = self.loss_fn(logits.reshape(B * S, C), labels.reshape(B * S))

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()

        return loss, correct, total

    @torch.no_grad()
    def _run_validation_loader(self, loader):
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in loader:
            if batch_is_empty(batch):
                continue

            loss, correct, n = self._forward_loss_and_metrics(batch)
            total_loss += float(loss.detach().cpu().item()) * n
            total_correct += correct
            total_n += n

        val_loss = total_loss / max(1, total_n)
        val_acc = total_correct / max(1, total_n)
        return val_loss, val_acc

    @torch.no_grad()
    def run_validation_pair(self):
        val_loss_noaug, val_acc_noaug = self._run_validation_loader(self.val_loader_noaug)
        val_loss_aug, val_acc_aug = self._run_validation_loader(self.val_loader_aug)

        return {
            "val_loss_noaug": val_loss_noaug,
            "val_acc_noaug": val_acc_noaug,
            "val_loss_aug": val_loss_aug,
            "val_acc_aug": val_acc_aug,
        }

    def _save_history_row(self, row):
        self.history_rows.append(row)

        write_header = not os.path.exists(self.history_path)
        with open(self.history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_validation_snapshot(self, *, stage_name, epoch_idx, epoch_progress, train_loss=None, train_acc=None, tag="epoch_end"):
        metrics = self.run_validation_pair()

        row = {
            "stage": stage_name,
            "epoch": int(epoch_idx),
            "epoch_progress": float(epoch_progress),
            "tag": tag,
            "train_loss": None if train_loss is None else float(train_loss),
            "train_acc": None if train_acc is None else float(train_acc),
            "val_loss_noaug": float(metrics["val_loss_noaug"]),
            "val_acc_noaug": float(metrics["val_acc_noaug"]),
            "val_loss_aug": float(metrics["val_loss_aug"]),
            "val_acc_aug": float(metrics["val_acc_aug"]),
        }
        self._save_history_row(row)

        if self.logger:
            wandb.log({
                "epoch": float(epoch_progress),
                f"{stage_name}/val_loss_noaug": metrics["val_loss_noaug"],
                f"{stage_name}/val_acc_noaug": metrics["val_acc_noaug"],
                f"{stage_name}/val_loss_aug": metrics["val_loss_aug"],
                f"{stage_name}/val_acc_aug": metrics["val_acc_aug"],
                "aggregate/val_loss_noaug": metrics["val_loss_noaug"],
                "aggregate/val_acc_noaug": metrics["val_acc_noaug"],
                "aggregate/val_loss_aug": metrics["val_loss_aug"],
                "aggregate/val_acc_aug": metrics["val_acc_aug"],
            })

        print(
            f"[{stage_name}][{tag} @ {epoch_progress:.3f}] "
            f"val_noaug_loss={metrics['val_loss_noaug']:.7f} "
            f"val_noaug_acc={metrics['val_acc_noaug']:.7f} | "
            f"val_aug_loss={metrics['val_loss_aug']:.7f} "
            f"val_aug_acc={metrics['val_acc_aug']:.7f}"
        )

        return metrics

    def run_epoch(self, stage_name, scheduler, epoch_idx, num_epochs):
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        num_batches = len(self.train_loader)

        # Example:
        # intra_epoch_val_points = 2 -> validate around 1/3 and 2/3 of epoch
        num_mid_vals = int(getattr(self.cfg.validation, "intra_epoch_val_points", 0))
        trigger_steps = set()
        if num_mid_vals > 0 and num_batches > 1:
            for k in range(1, num_mid_vals + 1):
                step = round(k * num_batches / (num_mid_vals + 1))
                step = min(max(step, 1), num_batches - 1)
                trigger_steps.add(step)

        progress = tqdm(self.train_loader, desc=f"{stage_name} | training", leave=False)

        for step_idx, batch in enumerate(progress, start=1):
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
            progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.7f}")

            if self.logger:
                wandb.log({
                    f"{stage_name}/loss_step": loss_val,
                    f"{stage_name}/acc_step": acc_val,
                    "aggregate/loss_step": loss_val,
                })

            # Mid-epoch validation
            if step_idx in trigger_steps:
                train_loss_so_far = total_loss / max(1, total_n)
                train_acc_so_far = total_correct / max(1, total_n)
                epoch_progress = epoch_idx + (step_idx / num_batches)

                self._log_validation_snapshot(
                    stage_name=stage_name,
                    epoch_idx=epoch_idx,
                    epoch_progress=epoch_progress,
                    train_loss=train_loss_so_far,
                    train_acc=train_acc_so_far,
                    tag=f"mid_epoch_step{step_idx}"
                )
                self.model.train()

        return (total_loss / max(1, total_n)), (total_correct / max(1, total_n))

    def run_stage(self, stage_dict):
        stage_name = stage_dict["name"]
        print(f"\nRunning stage: {stage_name}")
        print(">>> Optim cfg:", self.cfg.optim)

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
            train_loss, train_acc = self.run_epoch(
                stage_name=stage_name,
                scheduler=scheduler,
                epoch_idx=epoch,
                num_epochs=stage_dict["epochs"],
            )

            val_metrics = self._log_validation_snapshot(
                stage_name=stage_name,
                epoch_idx=epoch,
                epoch_progress=epoch + 1.0,
                train_loss=train_loss,
                train_acc=train_acc,
                tag="epoch_end",
            )

            val_loss = val_metrics["val_loss_noaug"]
            val_acc = val_metrics["val_acc_noaug"]

            print(
                f"[{stage_name}][Epoch {epoch}] "
                f"train_loss={train_loss:.7f} train_acc={train_acc:.7f} | "
                f"val_noaug_loss={val_metrics['val_loss_noaug']:.7f} "
                f"val_noaug_acc={val_metrics['val_acc_noaug']:.7f} | "
                f"val_aug_loss={val_metrics['val_loss_aug']:.7f} "
                f"val_aug_acc={val_metrics['val_acc_aug']:.7f}"
            )

            if self.logger:
                wandb.log({
                    "epoch_int": epoch,
                    f"{stage_name}/loss_epoch": train_loss,
                    f"{stage_name}/acc_epoch": train_acc,
                    f"{stage_name}/val_loss_epoch_noaug": val_metrics["val_loss_noaug"],
                    f"{stage_name}/val_acc_epoch_noaug": val_metrics["val_acc_noaug"],
                    f"{stage_name}/val_loss_epoch_aug": val_metrics["val_loss_aug"],
                    f"{stage_name}/val_acc_epoch_aug": val_metrics["val_acc_aug"],
                    "aggregate/val_loss_epoch_noaug": val_metrics["val_loss_noaug"],
                    "aggregate/val_acc_epoch_noaug": val_metrics["val_acc_noaug"],
                    "aggregate/val_loss_epoch_aug": val_metrics["val_loss_aug"],
                    "aggregate/val_acc_epoch_aug": val_metrics["val_acc_aug"],
                })

            improved = (val_acc > best_val_acc) or (
                (best_val_acc - val_acc < self.acc_epsilon) and (val_loss < best_val_loss)
            )

            if improved:
                best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
                torch.save(self.model.state_dict(), self.checkpoint_save_path)
                print(f"  -> saved best: val_acc={best_val_acc:.7f} val_loss={best_val_loss:.7f}")
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


@hydra.main(config_path="../configs", config_name="chess_classifier_vit_v2_stats", version_base="1.1")
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