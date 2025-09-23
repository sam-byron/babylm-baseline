# Example of how to integrate wandb with your existing training code

import wandb
import torch

def initialize_wandb(config, model, is_main_process=True):
    """Initialize wandb tracking."""
    if not is_main_process:
        return
    
    wandb.init(
        project="babylm-bert-training",
        name=f"bert_{config.get('experiment_name', 'default')}",
        config={
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"), 
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
            "max_grad_norm": config.get("max_grad_norm"),
            "num_epochs": config.get("num_epochs"),
            "model_type": model.__class__.__name__,
            "vocab_size": getattr(model.config, 'vocab_size', None),
            "hidden_size": getattr(model.config, 'hidden_size', None),
            "num_layers": getattr(model.config, 'num_hidden_layers', None),
        },
        tags=["bert", "babylm", "masked-lm"]
    )
    
    # Watch model for gradients and parameters
    wandb.watch(model, log="all", log_freq=100)


def log_training_metrics(loss, gradient_norm, learning_rate, step, epoch, 
                        monitor_stats=None, is_main_process=True):
    """Log training metrics to wandb."""
    if not is_main_process:
        return
        
    metrics = {
        "train/loss": float(loss),
        "train/learning_rate": float(learning_rate),
        "train/gradient_norm": float(gradient_norm),
        "train/step": step,
        "train/epoch": epoch,
    }
    
    # Add monitoring stats if available
    if monitor_stats:
        metrics.update({
            "monitor/spike_count": monitor_stats.get("spike_count", 0),
            "monitor/oscillation_count": monitor_stats.get("oscillation_count", 0), 
            "monitor/explosion_count": monitor_stats.get("explosion_count", 0),
            "monitor/loss_variance": monitor_stats.get("loss_variance", 0),
        })
    
    wandb.log(metrics)


def log_validation_metrics(val_loss, epoch, is_main_process=True):
    """Log validation metrics to wandb."""
    if not is_main_process:
        return
        
    wandb.log({
        "eval/validation_loss": float(val_loss),
        "eval/epoch": epoch,
    })


def log_blimp_results(blimp_results, epoch, is_main_process=True):
    """Log BLiMP evaluation results to wandb."""
    if not is_main_process:
        return
        
    if isinstance(blimp_results, dict):
        # Log overall accuracy
        overall_accuracy = blimp_results.get("overall_accuracy", 0)
        wandb.log({
            "eval/blimp_overall_accuracy": float(overall_accuracy),
            "eval/blimp_epoch": epoch,
        })
        
        # Log per-task accuracies
        task_metrics = {}
        for task, accuracy in blimp_results.items():
            if task != "overall_accuracy":
                task_metrics[f"eval/blimp_{task}"] = float(accuracy)
        
        if task_metrics:
            wandb.log(task_metrics)


def save_model_artifact(model, tokenizer, checkpoint_path, epoch, step, is_main_process=True):
    """Save model as wandb artifact."""
    if not is_main_process:
        return
        
    try:
        artifact = wandb.Artifact(
            name=f"bert_model_epoch_{epoch}_step_{step}",
            type="model",
            description=f"BERT model checkpoint at epoch {epoch}, step {step}"
        )
        
        artifact.add_dir(checkpoint_path)
        wandb.log_artifact(artifact)
        
        print(f"Saved model artifact to wandb: epoch {epoch}, step {step}")
        
    except Exception as e:
        print(f"Warning: Failed to save model artifact to wandb: {e}")


def finish_wandb_run():
    """Finish the wandb run."""
    try:
        wandb.finish()
    except:
        pass  # wandb might not be initialized


# Example usage in your training loop:
"""
# In main():
initialize_wandb(config, model, accelerator.is_main_process)

# In your training loop (replace existing monitor.update() calls):
log_training_metrics(
    loss=loss_value,
    gradient_norm=total_norm,
    learning_rate=current_lr,
    step=step,
    epoch=epoch,
    monitor_stats=monitor.get_stats(),
    is_main_process=is_main
)

# After validation:
log_validation_metrics(avg_val_loss, epoch, is_main)

# After BLiMP evaluation:
log_blimp_results(blimp_results, epoch, is_main)

# When saving checkpoints:
save_model_artifact(model, tokenizer, checkpoint_path, epoch, step, is_main)

# At the end:
finish_wandb_run()
"""