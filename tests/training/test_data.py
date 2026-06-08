from src.training.data import (
    apply_sliding_window, load_training_dataset, prepare_training_matrices, split_training_data,
)
from src.training.schema import TrainingConfig


def test_load_and_prepare(labeled_features):
    path, _ = labeled_features
    cfg = TrainingConfig(use_labels_for_evaluation=True, label_column="label")
    df, ctx, meta = load_training_dataset(path, cfg)
    assert ctx.row_count == len(df)
    x, y = prepare_training_matrices(df, cfg)
    assert x.shape[1] == 7 and y is not None
    xt, xv, yt, yv = split_training_data(x, y, cfg)
    assert len(xt) > len(xv)


def test_sliding_window_keeps_recent(labeled_features):
    _, df = labeled_features
    out = apply_sliding_window(df, window_days=1)
    # strict '<': the fixture spans ~25h, so a 1-day window MUST drop the
    # oldest rows. '<=' would pass even if the filter were a silent no-op.
    assert len(out) < len(df)
