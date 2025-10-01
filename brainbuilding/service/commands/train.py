from brainbuilding.service.pipeline import (
    PipelineConfig,
    load_pipeline_from_yaml,
)
from brainbuilding.core.utils import resolve_resource_path


import numpy as np
import typer
from typing import Optional, List, cast
from joblib import dump as joblib_dump
from sklearn.metrics import f1_score


import logging
import os


LOG_TRAIN = logging.getLogger(__name__)


def _load_and_calibrate_data(
    use_calibrated: bool,
    calibrated_array_path: str,
    sessions_dirs: Optional[List[str]],
    sessions_root: str,
    pipeline_config: PipelineConfig,
    state_config_path: str,
    save_calibrated: bool,
) -> np.ndarray:
    if use_calibrated:
        return np.load(calibrated_array_path, allow_pickle=True)

    from brainbuilding.service.eeg_service import EEGOfflineRunner

    if sessions_dirs is not None and len(sessions_dirs) > 0:
        session_dirs = [d for d in sessions_dirs if os.path.isdir(d)]
    else:
        session_dirs = [
            os.path.join(sessions_root, d)
            for d in os.listdir(sessions_root)
            if os.path.isdir(os.path.join(sessions_root, d))
        ]

    all_rows: List[np.ndarray] = []
    for sess_dir in sorted(session_dirs):
        sess_id = os.path.basename(sess_dir)
        xdf_path = os.path.join(sess_dir, "data.xdf")
        if not os.path.exists(xdf_path):
            logging.info("Skipping %s because it does not exist", xdf_path)
            continue
        logging.info("Loading %s", xdf_path)
        runner = EEGOfflineRunner(
            pipeline_config=pipeline_config,
            state_config_path=state_config_path,
        )
        rows = runner.run_from_xdf(xdf_path, session_id=sess_id)
        if rows:
            rows_arr = np.concatenate(cast(List[np.ndarray], rows))
            LOG_TRAIN.info(
                "Number of unique states within a subject %s: %d",
                sess_id,
                len(np.unique(rows_arr["state_visit_id"])),
            )
            all_rows.extend(cast(List[np.ndarray], rows))

    if not all_rows:
        raise RuntimeError(
            "No calibrated data produced from provided sessions"
        )

    data = np.concatenate(all_rows)
    if save_calibrated:
        np.save(calibrated_array_path, data)

    return data


def _evaluate_losocv(data: np.ndarray, training_cfg: PipelineConfig) -> None:
    y = data["label"]
    scores = []
    for session_id in np.unique(data["session_id"]):
        train_mask = data["session_id"] != session_id
        test_mask = data["session_id"] == session_id

        X_train, y_train = data[train_mask], y[train_mask]
        X_test, y_test = data[test_mask], y[test_mask]

        fitted = training_cfg.fit_training(X_train, y_train)

        y_preds: List[np.ndarray] = []
        skipped_samples: List[int] = []
        for i in range(len(X_test)):
            sample = X_test[i : i + 1]
            try:
                fitted.update(
                    training_cfg.partial_fit_components(sample, fitted)
                )
                y_pred = training_cfg.apply_training(sample, fitted)[:, 0]
                y_preds.append(y_pred)
            except ValueError as e:
                LOG_TRAIN.error(
                    "Skipping sample %d for session %s: %s", i, session_id, e
                )
                skipped_samples.append(i)
                continue

        y_preds_arr = np.array(y_preds).flatten()
        mask = np.ones(len(y_test), dtype=bool)
        if skipped_samples:
            mask[skipped_samples] = False

        score = f1_score(y_test[mask], y_preds_arr)

        LOG_TRAIN.info(
            "F1 score for session %s during LOSOCV: %.4f", session_id, score
        )
        scores.append(score)

    (
        min_score,
        max_score,
        median_score,
        q1_score,
        q3_score,
        mean_score,
        std_score,
    ) = (
        np.min(scores),
        np.max(scores),
        np.median(scores),
        np.percentile(scores, 25),
        np.percentile(scores, 75),
        np.mean(scores),
        np.std(scores),
    )
    logging.info(
        (
            "LOSOCV F1 stats: min=%.4f, max=%.4f, median=%.4f, "
            "q1=%.4f, q3=%.4f, mean=%.4f, std=%.4f"
        ),
        min_score,
        max_score,
        median_score,
        q1_score,
        q3_score,
        mean_score,
        std_score,
    )


def _train_and_save_model(
    data: np.ndarray, training_cfg: PipelineConfig, output_dir: str
) -> None:
    y = data["label"]
    fitted = training_cfg.fit_training(data, y)
    os.makedirs(output_dir, exist_ok=True)
    for name, comp in fitted.items():
        outp = os.path.join(output_dir, f"{name}.joblib")
        joblib_dump(comp, outp)
    logging.info("Saved %d components to %s", len(fitted), output_dir)


def train(
    calibrated_array_path: str = typer.Option(
        "data/calibrated.npy", help="Path to save/load calibrated array"
    ),
    save_calibrated: bool = typer.Option(
        True, help="Save calibrated array to disk"
    ),
    use_calibrated: bool = typer.Option(
        False, help="Use pre-saved calibrated array"
    ),
    evaluate_losocv: bool = typer.Option(False, help="Evaluate using LOSOCV"),
    sessions_root: str = typer.Option(
        "data", help="Root directory containing session subfolders"
    ),
    sessions_dirs: Optional[List[str]] = typer.Option(
        None, help="Explicit list of session directories"
    ),
    output_dir: str = typer.Option(
        "models", help="Output directory for trained components"
    ),
    exclude_label: int = typer.Option(
        2, help="Exclude samples with this label (-1 to disable)"
    ),
    pipeline_config_path: str = typer.Option(
        help="Pipeline YAML path for training",
    ),
    state_config_path: str = typer.Option(
        help="State machine YAML path",
    ),
    preload_dir: str = typer.Option(
        "models", help="Preload directory for components"
    ),
    no_preload: bool = typer.Option(
        True, help="Disable preloading components"
    ),
) -> None:
    logging.basicConfig(level=logging.INFO)

    resolved_pipeline_path = resolve_resource_path(pipeline_config_path)
    resolved_state_path = resolve_resource_path(state_config_path)
    resolved_output_dir = resolve_resource_path(output_dir)
    resolved_preload_dir = (
        resolve_resource_path(preload_dir) if not no_preload else None
    )

    pipeline_config, _ = load_pipeline_from_yaml(
        resolved_pipeline_path,
        preload_dir=resolved_preload_dir,
        enable_preload=not no_preload,
    )

    all_rows_arr = _load_and_calibrate_data(
        use_calibrated,
        calibrated_array_path,
        sessions_dirs,
        sessions_root,
        pipeline_config,
        resolved_state_path,
        save_calibrated,
    )

    logging.info("all_rows_arr.shape=%s", all_rows_arr.shape)
    if all_rows_arr.dtype.names and "sample" in all_rows_arr.dtype.names:
        logging.info(
            "all_rows_arr['sample'].shape=%s", all_rows_arr["sample"].shape
        )

    training_cfg = pipeline_config.training_only_config()

    if exclude_label != -1:
        all_rows_arr = all_rows_arr[all_rows_arr["label"] != exclude_label]

    if evaluate_losocv:
        _evaluate_losocv(all_rows_arr, training_cfg)

    _train_and_save_model(all_rows_arr, training_cfg, resolved_output_dir)
