import argparse
import math
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from prepare_blood_cell_cancer_data import (
    BATCH_SIZE,
    IMG_SHAPE,
    SEED,
    SplitPaths,
    build_best_classification_datasets,
    set_seed,
)


def load_classes(metadata_path: Path) -> Sequence[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["classes"]


def load_split_sizes(metadata_path: Path) -> Dict[str, int]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["split_sizes"]


def load_split_paths(prepared_dir: Path) -> SplitPaths:
    tfrecords_dir = prepared_dir / "tfrecords"
    return SplitPaths(
        train=tfrecords_dir / "blood_cell_cancer_with_mask_train.tfrecord",
        val=tfrecords_dir / "blood_cell_cancer_with_mask_val.tfrecord",
        test=tfrecords_dir / "blood_cell_cancer_with_mask_test.tfrecord",
    )


def get_backbone(img_shape):
    backbone = VGG16(include_top=False, weights="imagenet", input_shape=img_shape + (3,), pooling="avg")
    backbone.trainable = False
    return backbone


def classification_with_backbone_architecture(inp, num_classes: int):
    backbone = get_backbone(IMG_SHAPE)
    x = backbone(inp, training=False)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)
    return x


def get_model(num_classes: int) -> Model:
    tf.keras.backend.clear_session()
    inp = Input(shape=IMG_SHAPE + (3,))
    output = classification_with_backbone_architecture(inp, num_classes)
    model = Model(inputs=inp, outputs=output)
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=1e-3),
        metrics=[AUC(name="auc")],
    )
    return model


def compute_class_weights(train_csv: Path, class_names: Sequence[str]) -> Dict[int, float]:
    train_df = pd.read_csv(train_csv)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_names),
        y=train_df["type_cell"],
    )
    return dict(zip(range(len(class_names)), weights))


def get_callbacks(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3),
        ModelCheckpoint(
            filepath=str(checkpoints_dir / "model_backbone.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        CSVLogger(str(output_dir / "history_backbone.csv"), separator=","),
    ]


def get_scores(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr")),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
    }


def evaluate_model(model: Model, test_dataset: tf.data.Dataset, test_steps: int) -> Dict[str, float]:
    keras_metrics = model.evaluate(test_dataset, steps=test_steps, return_dict=True, verbose=1)

    y_true_batches = []
    for _, class_ohe in test_dataset:
        y_true_batches.append(class_ohe.numpy())
    y_true = np.concatenate(y_true_batches, axis=0)

    y_pred_proba = model.predict(test_dataset, steps=test_steps, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)

    scores = get_scores(y_true_idx, y_pred, y_pred_proba)
    scores["test_loss"] = float(keras_metrics["loss"])
    scores["test_auc"] = float(keras_metrics["auc"])
    return scores


def train(prepared_dir: Path, output_dir: Path, epochs: int, batch_size: int, seed: int) -> Dict[str, float]:
    set_seed(seed)

    metadata_path = prepared_dir / "metadata.json"
    train_csv = prepared_dir / "metadata" / "train.csv"
    class_names = load_classes(metadata_path)
    split_sizes = load_split_sizes(metadata_path)
    class_weight = compute_class_weights(train_csv, class_names)
    datasets = build_best_classification_datasets(
        tfrecord_paths=load_split_paths(prepared_dir),
        class_names=class_names,
        batch_size=batch_size,
        seed=seed,
    )
    train_steps = math.ceil(split_sizes["train"] / batch_size)
    val_steps = math.ceil(split_sizes["val"] / batch_size)
    test_steps = math.ceil(split_sizes["test"] / batch_size)

    model = get_model(num_classes=len(class_names))
    history = model.fit(
        datasets["train"].repeat(),
        epochs=epochs,
        callbacks=get_callbacks(output_dir),
        validation_data=datasets["val"].repeat(),
        class_weight=class_weight,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=1,
    )

    final_model_path = output_dir / "model_backbone_final.keras"
    model.save(final_model_path)

    results = evaluate_model(model, datasets["test"], test_steps)
    results["epochs_ran"] = len(history.history["loss"])
    results["class_weight"] = {str(k): float(v) for k, v in class_weight.items()}
    results["classes"] = list(class_names)
    results["model_path"] = str(final_model_path.resolve())

    (output_dir / "evaluation_backbone.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de clasificacion VGG16 usando los TFRecords preparados de blood cell cancer."
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts") / "blood_cell_cancer_prepared",
        help="Directorio creado por prepare_blood_cell_cancer_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "blood_cell_cancer_training" / "backbone_vgg16",
        help="Directorio para historial, checkpoints, modelo final y metricas.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero maximo de epocas.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size para entrenamiento y evaluacion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla de reproducibilidad.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = train(
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
