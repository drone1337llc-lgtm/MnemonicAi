# --- Add near the top of start.py ---
try:
    from app.metrics import setup_metrics, update_adapter_metrics
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# --- After you create the FastAPI app instance ---
if METRICS_ENABLED:
    setup_metrics(app)

# --- After training completes ---
if METRICS_ENABLED:
    from app.metrics import TRAINING_EVENTS, TRAINING_LOSS, EVAL_LOSS
    TRAINING_EVENTS.labels(result="success").inc()
    TRAINING_LOSS.set(result["loss"])
    if "eval_loss" in result:
        EVAL_LOSS.set(result["eval_loss"])
    update_adapter_metrics(os.path.join(data_dir, "adapter"))
