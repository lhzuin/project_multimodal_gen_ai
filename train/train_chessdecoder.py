import math
import os
import random
import warnings

import numpy as np
import torch
import hydra
import chess
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from transformers.optimization import get_scheduler
from tqdm.auto import tqdm

from train.training_utils import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    GracefulStop,
    TrainingInterrupted,
    capture_rng_state,
    init_wandb,
    model_weights,
    restore_rng_state,
    select_device,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b, replace=True)


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
    if not isinstance(batch, dict):
        return True
    if batch.get("valid") is False:
        return True
    if "input_ids" in batch and batch["input_ids"].numel() > 0:
        return False
    return True


def build_legal_mask_from_fen(fen_list):
    """
    fen_list: list[str] length B
    returns mask [B,64,64] bool where mask[b,from,to]=True iff legal
    """
    B = len(fen_list)
    mask = torch.zeros((B, 64, 64), dtype=torch.bool)
    for i, fen in enumerate(fen_list):
        board = chess.Board(fen)
        for mv in board.legal_moves:
            mask[i, mv.from_square, mv.to_square] = True
    return mask

def apply_legal_mask(policy_logits, legal_mask):
    # policy_logits: [B,64,64]
    # legal_mask:    [B,64,64] bool
    neg_inf = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~legal_mask.to(policy_logits.device), neg_inf)


def move_metrics(from_logits, to_logits, move_from, move_to):
    # accuracy on from/to separately
    from_pred = from_logits.argmax(dim=-1)
    to_pred = to_logits.argmax(dim=-1)
    acc_from = (from_pred == move_from).float().mean().item()
    acc_to = (to_pred == move_to).float().mean().item()
    return acc_from, acc_to


class Trainer:
    def __init__(self, cfg, logger, device, stop_controller=None):
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
        self.scheduler = None
        self.stage_name = None
        self.stage_index = 0
        self.current_epoch = 0
        self.next_batch_idx = 0
        self.global_step = 0
        self.epoch_generator_state = None
        self.stop_controller = stop_controller or GracefulStop()
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        self.patience = cfg.early_stopping.patience
        self.min_epochs = cfg.early_stopping.min_epochs
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

        self.grad_accum_steps = int(getattr(cfg, "trainer", {}).get("grad_accum_steps", 1))
        self.grad_clip_norm = float(getattr(cfg, "trainer", {}).get("grad_clip_norm", 1.0))
        self.checkpoint_load_path = getattr(cfg, "checkpoint_load_path", None)
        self.train_generator = torch.Generator().manual_seed(int(cfg.seed))

        keep_last = int(getattr(cfg, "checkpoint_keep_last", 3))
        self.checkpoints = CheckpointManager(cfg.checkpoint_save_path, keep_last=keep_last)
        resume_path = self.checkpoints.resolve_resume_path(self.checkpoint_load_path)
        self.resume_checkpoint = self.checkpoints.load(resume_path) if resume_path else None
        self.resume_applied = False

    @property
    def wandb_resume_id(self):
        if isinstance(self.resume_checkpoint, dict):
            return self.resume_checkpoint.get("wandb_run_id")
        return None

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
            pin_memory=bool(dl_cfg.pin_memory and self.device.type == "cuda"),
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=getattr(dl_cfg, "drop_last", False),
            generator=self.train_generator,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=False,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(dl_cfg.pin_memory and self.device.type == "cuda"),
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=False,
        )

        print(f"Dataset size={len(self.train_dataset)} | train={len(train_ds)} | val={len(val_ds)}")

    def init_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model.instance).to(self.device)
        self.loss_fn = hydra.utils.instantiate(self.cfg.loss_fn)
        if self.resume_checkpoint is not None and not (
            isinstance(self.resume_checkpoint, dict) and "optimizer" in self.resume_checkpoint
        ):
            self.model.load_state_dict(model_weights(self.resume_checkpoint))
            self.resume_applied = True
            print("Loaded a weight-only checkpoint; optimizer and epoch start from scratch.")

    
    def _forward_loss_and_metrics(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        piece_labels = batch["piece_labels"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        start_turn = batch["start_turn"].to(self.device, non_blocking=True)

        piece_input_ids = batch["piece_input_ids"].to(self.device, non_blocking=True)

        out = self.model(
            input_ids=input_ids,
            piece_input_ids=piece_input_ids,
            attention_mask=attention_mask,
            start_turn=start_turn,
        )
        logits = out["logits"]              # [B,T,Vmove]
        piece_logits = out["piece_logits"]  # [B,T,Vpiece]

        loss_move = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss_piece = F.cross_entropy(piece_logits.view(-1, piece_logits.size(-1)), piece_labels.view(-1), ignore_index=-100)

        lam = float(getattr(self.cfg, "piece_loss_weight", 0.3))
        loss = loss_move + lam * loss_piece
        if not torch.isfinite(loss):
            return None
        

        # token-accuracy on move labels (policy quality proxy)
        with torch.no_grad():
            # logits: [B,T,V]
            preds = logits.argmax(dim=-1)  # [B,T]
            m = (labels != -100)           # [B,T] boolean mask of valid targets
            n = int(m.sum().item())

            if n > 0:
                # top-1 correct count
                correct1 = (preds[m] == labels[m]).sum().item()

                # top-5 correct count
                # topk_idx: [B,T,5]
                topk_idx = torch.topk(logits, k=5, dim=-1).indices
                # compare labels with top-5 predictions at each position
                correct5 = (topk_idx[m] == labels[m].unsqueeze(-1)).any(dim=-1).sum().item()
            else:
                correct1, correct5 = 0, 0

        return loss, correct1, correct5, n

    def run_epoch(self, stage_name, start_batch_idx=0):
        self.model.train()
        total_loss, total_correct, total_correct5, total_n = 0.0, 0, 0, 0

        accum = 0
        self.optimizer.zero_grad(set_to_none=True)

        self.epoch_generator_state = self.train_generator.get_state()
        progress = tqdm(self.train_loader, desc=f"{stage_name} | training", leave=False)
        for batch_idx, batch in enumerate(progress):
            if batch_idx < start_batch_idx:
                continue
            if self.stop_controller.requested and accum == 0:
                self.next_batch_idx = batch_idx
                raise TrainingInterrupted
            if batch_is_empty(batch):
                continue

            if self.scaler:
                with autocast(device_type=self.device.type):
                    res = self._forward_loss_and_metrics(batch)
                    if res is None:
                        continue
                    loss, correct, correct5, n = res
                    loss_scaled = loss / self.grad_accum_steps

                self.scaler.scale(loss_scaled).backward()
            else:
                res = self._forward_loss_and_metrics(batch)
                if res is None:
                    continue
                loss, correct, correct5, n = res
                (loss / self.grad_accum_steps).backward()

            accum += 1

            # --- logging stays per *micro-batch* exactly like before ---
            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val * n
            total_correct += correct
            total_correct5 += correct5
            total_n += n

            acc_val = correct / max(1, n)
            acc5_val = correct5 / max(1, n)
            progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.3f}", acc5=f"{acc5_val:.3f}")

            if self.logger:
                self.logger.log({
                    f"{stage_name}/loss_step": loss_val,
                    f"{stage_name}/acc_step": acc_val,
                    f"{stage_name}/acc5_step": acc5_val,
                    "aggregate/loss_step": loss_val,
                })

            # --- optimizer/scheduler step only every grad_accum_steps ---
            if accum % self.grad_accum_steps == 0:
                if self.scaler:
                    # IMPORTANT: unscale before clipping when using GradScaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler:
                    self.scheduler.step()

                self.global_step += 1
                self.next_batch_idx = batch_idx + 1
                if self.stop_controller.requested:
                    raise TrainingInterrupted

        # If last partial accumulation didn't trigger a step, you can either:
        # (A) ignore it (common), or
        # (B) step once more to not waste gradients.
        # Minimal behavior: DO NOT step here (keeps semantics stable).

        return (total_loss / max(1, total_n)), (total_correct / max(1, total_n)), (total_correct5 / max(1, total_n))

    @torch.no_grad()
    def run_validation(self):
        self.model.eval()
        total_loss, total_correct, total_correct5, total_n = 0.0, 0, 0, 0

        for batch in self.val_loader:
            if self.stop_controller.requested:
                raise TrainingInterrupted
            if batch_is_empty(batch):
                continue

            res = self._forward_loss_and_metrics(batch)
            if res is None:
                continue
            loss, correct, correct5, n = res
            total_loss += float(loss.detach().cpu().item()) * n
            total_correct += correct
            total_correct5 += correct5
            total_n += n

        val_loss = total_loss / max(1, total_n)
        val_acc = total_correct / max(1, total_n)
        val_acc5 = total_correct5 / max(1, total_n)
        return val_loss, val_acc, val_acc5

    def _restore_training_state(self, stage_name, stage_index):
        checkpoint = self.resume_checkpoint
        if self.resume_applied or not isinstance(checkpoint, dict) or "optimizer" not in checkpoint:
            return 0, 0, float("inf"), 0.0, 0, False

        progress = checkpoint.get("progress", {})
        saved_stage = progress.get("stage_name", checkpoint.get("stage_name"))
        saved_stage_index = int(progress.get("stage_index", stage_index))
        if saved_stage != stage_name or saved_stage_index != stage_index:
            return 0, 0, float("inf"), 0.0, 0, False

        self.model.load_state_dict(model_weights(checkpoint))
        if checkpoint.get("optimizer") is None:
            raise ValueError("The selected checkpoint has no optimizer state and cannot be resumed")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        restore_rng_state(checkpoint.get("rng_state"))
        if checkpoint.get("train_generator_state") is not None:
            self.train_generator.set_state(checkpoint["train_generator_state"])
        self.global_step = int(progress.get("global_step", checkpoint.get("global_step", 0)))

        epoch = int(progress.get("epoch", checkpoint.get("epoch", -1)))
        epoch_complete = bool(progress.get("epoch_complete", True))
        start_epoch = epoch + 1 if epoch_complete else max(0, epoch)
        start_batch = 0 if epoch_complete else int(progress.get("next_batch_idx", 0))
        early = checkpoint.get("early_stopping", {})
        self.best_val_loss = float(early.get("best_val_loss", float("inf")))
        self.best_val_acc = float(early.get("best_val_acc", 0.0))
        self.epochs_no_improve = int(early.get("epochs_no_improve", 0))
        self.resume_applied = True
        print(f"Resuming {stage_name} from epoch {start_epoch}, batch {start_batch}")
        return (
            start_epoch,
            start_batch,
            self.best_val_loss,
            self.best_val_acc,
            self.epochs_no_improve,
            bool(progress.get("stage_complete", False)),
        )

    def _checkpoint_state(self, *, epoch_complete, stage_complete, early_stopping):
        generator_state = self.train_generator.get_state()
        if not epoch_complete and self.epoch_generator_state is not None:
            generator_state = self.epoch_generator_state
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "training_state",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "progress": {
                "stage_name": self.stage_name,
                "stage_index": self.stage_index,
                "epoch": self.current_epoch,
                "next_batch_idx": 0 if epoch_complete else self.next_batch_idx,
                "epoch_complete": epoch_complete,
                "stage_complete": stage_complete,
                "global_step": self.global_step,
            },
            "early_stopping": dict(early_stopping),
            "rng_state": capture_rng_state(),
            "train_generator_state": generator_state,
            "config": OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True),
            "wandb_run_id": self.logger.id if self.logger else self.wandb_resume_id,
            "stage_name": self.stage_name,
            "epoch": self.current_epoch,
        }

    def save_checkpoint(self, *, epoch_complete, stage_complete=False, early_stopping=None, snapshot=False):
        if self.model is None or self.optimizer is None or self.stage_name is None:
            return None
        early_stopping = early_stopping or {
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "epochs_no_improve": self.epochs_no_improve,
        }
        state = self._checkpoint_state(
            epoch_complete=epoch_complete,
            stage_complete=stage_complete,
            early_stopping=early_stopping,
        )
        path = self.checkpoints.save_latest(
            state,
            self.model.state_dict(),
            stage_name=self.stage_name,
            epoch=self.current_epoch,
            save_snapshot=snapshot,
        )
        print(f"Saved resumable checkpoint -> {path}")
        return path

    def run_stage(self, stage_dict, stage_index):
        stage_name = stage_dict["name"]
        self.stage_name = stage_name
        self.stage_index = stage_index
        print(f"\nRunning stage: {stage_name}")
        print(">>> Optim cfg:", self.cfg.optim)

        # Freeze/unfreeze as before
        self.model.set_trainable_groups(list(stage_dict["groups"].keys()))

        self.optimizer = hydra.utils.instantiate(
            self.opt_cfg,
            params=param_groups_for(self.model, self.cfg, stage_dict),
            _convert_="all",
        )

        self.scheduler = None
        if self.cfg.use_warmup:
            num_training_steps = stage_dict["epochs"] * math.ceil(len(self.train_loader) / self.grad_accum_steps)
            self.scheduler = build_scheduler_for_stage(
                self.optimizer,
                stage_dict=stage_dict,
                cfg=self.cfg,
                num_training_steps=num_training_steps,
            )

        (
            start_epoch,
            start_batch,
            best_val_loss,
            best_val_acc,
            epochs_no_improve,
            stage_complete,
        ) = self._restore_training_state(stage_name, stage_index)
        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_acc
        self.epochs_no_improve = epochs_no_improve
        if stage_complete:
            print(f"Stage {stage_name} was already complete; skipping it.")
            return

        for epoch in range(start_epoch, stage_dict["epochs"]):
            self.current_epoch = epoch
            self.next_batch_idx = start_batch if epoch == start_epoch else 0
            train_loss, train_acc, train_acc5 = self.run_epoch(
                stage_name, start_batch_idx=self.next_batch_idx
            )
            if self.stop_controller.requested:
                raise TrainingInterrupted
            start_batch = 0
            val_loss, val_acc, val_acc5 = self.run_validation()

            print(
                f"[{stage_name}][Epoch {epoch}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_acc5={train_acc5:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_acc5={val_acc5:.4f}"
            )

            if self.logger:
                self.logger.log({
                    "epoch": epoch,
                    f"{stage_name}/loss_epoch": train_loss,
                    f"{stage_name}/acc_epoch": train_acc,
                    f"{stage_name}/acc5_epoch": train_acc5,
                    f"{stage_name}/val_loss_epoch": val_loss,
                    f"{stage_name}/val_acc_epoch": val_acc,
                    f"{stage_name}/val_acc5_epoch": val_acc5,
                    "aggregate/val_loss_epoch": val_loss,
                    "aggregate/val_acc_epoch": val_acc,
                    "aggregate/val_acc5_epoch": val_acc5,
                })

            improved = (val_acc > best_val_acc) or (
                (best_val_acc - val_acc < self.acc_epsilon) and (val_loss < best_val_loss)
            )

            if improved:
                best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
                best_path = self.checkpoints.save_best(self.model.state_dict())
                print(
                    f"  -> saved best weights to {best_path}: "
                    f"val_acc={best_val_acc:.4f} val_loss={best_val_loss:.4f}"
                )
            else:
                epochs_no_improve += 1
            self.best_val_loss = best_val_loss
            self.best_val_acc = best_val_acc
            self.epochs_no_improve = epochs_no_improve
            should_stop = epochs_no_improve >= self.patience and epoch >= self.min_epochs
            stage_complete = should_stop or epoch + 1 >= int(stage_dict["epochs"])
            self.save_checkpoint(
                epoch_complete=True,
                stage_complete=stage_complete,
                early_stopping={
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "epochs_no_improve": epochs_no_improve,
                },
                snapshot=True,
            )
            if should_stop:
                print(f"Early stopping {stage_name}.")
                break

    def run_train(self):
        resume_progress = (
            self.resume_checkpoint.get("progress", {})
            if isinstance(self.resume_checkpoint, dict)
            else {}
        )
        resume_stage_index = int(resume_progress.get("stage_index", 0))
        if self.resume_checkpoint is not None and not self.resume_applied:
            if not 0 <= resume_stage_index < len(self.cfg.train_stages):
                raise ValueError(
                    f"Checkpoint stage index {resume_stage_index} is absent from the active config"
                )
            saved_name = resume_progress.get(
                "stage_name", self.resume_checkpoint.get("stage_name")
            )
            expected_name = self.cfg.train_stages[resume_stage_index]["name"]
            if saved_name != expected_name:
                raise ValueError(
                    f"Checkpoint stage {saved_name!r} does not match configured stage {expected_name!r}"
                )

        for i, stage_dict in enumerate(self.cfg.train_stages):
            if not self.resume_applied and i < resume_stage_index:
                continue
            if i > 0 and not (not self.resume_applied and i == resume_stage_index):
                print("Reloading best model from previous stage.")
                self.model.load_state_dict(self.checkpoints.load_best_weights())
            self.run_stage(stage_dict, i)


@hydra.main(config_path="../configs", config_name="chess_decoder_player_v12", version_base="1.1")
def main(cfg):
    set_seed(int(cfg.seed))

    device = select_device(getattr(cfg, "device", "auto"))
    print(f"Device={device} | PID={os.getpid()}  (kill -SIGUSR1 {os.getpid()} to checkpoint+exit)")

    stop_controller = GracefulStop()
    trainer = Trainer(cfg=cfg, logger=None, device=device, stop_controller=stop_controller)
    trainer.init_data()
    trainer.init_model()
    trainer.logger = init_wandb(cfg, resume_id=trainer.wandb_resume_id)
    stop_controller.install()
    try:
        trainer.run_train()
    except TrainingInterrupted:
        trainer.save_checkpoint(epoch_complete=False)
        print("Training stopped cleanly; resume with checkpoint_load_path=auto")
    except Exception:
        try:
            trainer.save_checkpoint(epoch_complete=False)
        except Exception as checkpoint_error:
            warnings.warn(f"Emergency checkpoint failed: {checkpoint_error}", RuntimeWarning)
        raise
    finally:
        stop_controller.restore()
        if trainer.logger:
            trainer.logger.finish()


if __name__ == "__main__":
    main()
