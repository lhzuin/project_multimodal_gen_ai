"""Auxiliary-board variant of the shared decoder training pipeline."""

from collections import defaultdict
import os
import warnings

import hydra
import torch

from train.decoder_board_objective import DecoderBoardObjective
from train.train_chessdecoder import Trainer, set_seed
from train.training_utils import GracefulStop, TrainingInterrupted, init_wandb, select_device


class BoardTrainer(Trainer):
    """Override the batch objective and component reporting in a separate trainer.

    Data loading, optimization, mixed precision, scheduling, early stopping,
    checkpoint rotation, interruptions and exact resume use the base trainer.
    """

    def __init__(self, cfg, logger, device, stop_controller=None):
        super().__init__(cfg, logger, device, stop_controller)
        self.objective = DecoderBoardObjective(
            board_loss_weight=float(cfg.auxiliary_board.loss_weight),
            piece_loss_weight=float(cfg.piece_loss_weight),
            normalization=str(cfg.auxiliary_board.normalization),
        )
        if self.objective.board_loss_weight > 0:
            for stage in cfg.train_stages:
                if "board_head" not in stage.groups:
                    raise ValueError("Each auxiliary training stage must optimize the board_head group")
        self._reset_components()

    def _forward_loss_and_metrics(self, batch):
        """Compute the variant's objective and the original move-accuracy counts.

        This is the batch boundary used by the inherited epoch/validation loops.
        The original decoder trainer requires no auxiliary-specific hooks.
        """
        labels = batch["labels"].to(self.device, non_blocking=True)
        active = labels != -100
        n = int(active.sum().item())
        if n == 0:
            return None
        inputs = {
            key: batch[key].to(self.device, non_blocking=True)
            for key in ("input_ids", "piece_input_ids", "attention_mask", "start_turn")
        }
        piece_labels = batch["piece_labels"].to(self.device, non_blocking=True)
        out = self._forward_model(**inputs)
        loss = self._compute_loss(out, labels, piece_labels, batch)
        if not torch.isfinite(loss):
            return None
        with torch.no_grad():
            logits = out["logits"]
            correct1 = (logits.argmax(-1)[active] == labels[active]).sum().item()
            top5 = torch.topk(logits, k=5, dim=-1).indices
            correct5 = (top5[active] == labels[active].unsqueeze(-1)).any(-1).sum().item()
        return loss, correct1, correct5, n

    def _forward_model(self, **inputs):
        # Validation puts the model in eval mode but still measures the auxiliary
        # objective. Normal inference leaves this flag unset and skips the head.
        return self.model(**inputs, return_board_logits=self.objective.board_loss_weight > 0)

    def _compute_loss(self, out, labels, piece_labels, batch):
        board_labels = batch["board_labels"].to(self.device, non_blocking=True)
        components = self.objective(out, labels, piece_labels, board_labels)
        loss = components["loss"]
        if not torch.isfinite(loss):
            return loss
        with torch.no_grad():
            n = int((labels != -100).sum().item())
            # Match the base trainer's move-token weighting for loss reports.
            names = list(components)
            values = torch.stack([components[name].detach() for name in names]).cpu().tolist()
            for name, value in zip(names, values):
                self._component_sums[name] += value * n
            self._component_count += n
            if "board_logits" in out:
                active = (board_labels != -100) & (labels != -100).unsqueeze(-1)
                correct = out["board_logits"].argmax(-1) == board_labels
                occupied = active & (board_labels != 0)
                valid_boards = active.all(-1)
                self._board_counts["square_correct"] += int((correct & active).sum().item())
                self._board_counts["square_total"] += int(active.sum().item())
                self._board_counts["occupied_correct"] += int((correct & occupied).sum().item())
                self._board_counts["occupied_total"] += int(occupied.sum().item())
                self._board_counts["board_correct"] += int((correct.all(-1) & valid_boards).sum().item())
                self._board_counts["board_total"] += int(valid_boards.sum().item())
        return loss

    def _reset_components(self):
        self._component_sums = defaultdict(float)
        self._component_count = 0
        self._board_counts = defaultdict(int)

    def _report_components(self, prefix):
        metrics = {
            f"{prefix}/{name}": value / max(1, self._component_count)
            for name, value in self._component_sums.items()
        }
        for metric, key in (("square_acc", "square"), ("occupied_square_acc", "occupied"), ("exact_board_acc", "board")):
            metrics[f"{prefix}/{metric}"] = self._board_counts[f"{key}_correct"] / max(1, self._board_counts[f"{key}_total"])
        if self.logger:
            self.logger.log(metrics)
        print(f"{prefix}: " + " ".join(f"{key.rsplit('/', 1)[-1]}={value:.4f}" for key, value in metrics.items()))

    def run_epoch(self, stage_name, start_batch_idx=0):
        self._reset_components()
        result = super().run_epoch(stage_name, start_batch_idx=start_batch_idx)
        self._report_components(f"{stage_name}/board_train")
        return result

    def run_validation(self):
        self._reset_components()
        result = super().run_validation()
        self._report_components(f"{self.stage_name}/board_val")
        return result


@hydra.main(config_path="../configs", config_name="chess_decoder_player_balanced_board_v1", version_base="1.1")
def main(cfg):
    """Run the auxiliary variant from scratch, or explicitly resume its checkpoint.

    Keep the entry point separate from the original trainer while using the same
    runtime utilities and inherited training/checkpoint lifecycle.
    """
    set_seed(int(cfg.seed))
    device = select_device(getattr(cfg, "device", "auto"))
    print(f"Device={device} | PID={os.getpid()}  (kill -SIGUSR1 {os.getpid()} to checkpoint+exit)")
    stop_controller = GracefulStop()
    trainer = BoardTrainer(cfg=cfg, logger=None, device=device, stop_controller=stop_controller)
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
