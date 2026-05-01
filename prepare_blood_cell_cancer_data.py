import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 42
IMG_SHAPE = (192, 256)
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 1 - (7 / 8.5)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_classes(raw_dir: Path) -> List[str]:
    return [item.name for item in raw_dir.iterdir() if item.is_dir()]


def infer_class(path: Path, classes: Sequence[str]) -> str:
    path_text = str(path)
    for class_name in classes:
        if class_name in path_text:
            return class_name
    raise ValueError(f"No se pudo inferir la clase para: {path}")


def build_image_dataframe(raw_dir: Path, classes: Sequence[str]) -> pd.DataFrame:
    records = []
    for image_path in raw_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            {
                "pathfiles": str(image_path.resolve()),
                "type_cell": infer_class(image_path, classes),
                "width": width,
                "heigth": height,
            }
        )

    if not records:
        raise FileNotFoundError(f"No se encontraron imagenes dentro de {raw_dir}")

    return pd.DataFrame(records)


def read_image(pathfile: str) -> np.ndarray:
    image = cv2.imread(pathfile)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {pathfile}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    resized = tf.image.resize(image, size=img_shape, method="nearest", antialias=True).numpy()
    return resized.astype(np.uint8)


def rgb_to_lab(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    return l_channel, a_channel, b_channel


def get_mask(image: np.ndarray) -> np.ndarray:
    _, a_channel, _ = rgb_to_lab(image)
    a_blur = cv2.GaussianBlur(a_channel, (19, 19), 0)
    _, thresh_img = cv2.threshold(
        a_blur,
        200,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(thresh_img, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    return mask.astype(np.uint8)


def split_dataframe(df_imgs: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    df_full_train, df_test = train_test_split(
        df_imgs,
        test_size=TEST_SIZE,
        stratify=df_imgs["type_cell"],
        shuffle=True,
        random_state=seed,
    )
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=VAL_SIZE_FROM_TRAIN,
        stratify=df_full_train["type_cell"],
        shuffle=True,
        random_state=seed,
    )
    return {"train": df_train.reset_index(drop=True), "val": df_val.reset_index(drop=True), "test": df_test.reset_index(drop=True)}


def bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(pathfile: Path, dataset: pd.DataFrame, classes: Sequence[str], img_shape: Tuple[int, int]) -> int:
    pathfile.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with tf.io.TFRecordWriter(str(pathfile)) as writer:
        for _, row in dataset.iterrows():
            image = resize_image(read_image(row["pathfiles"]), img_shape)
            mask = get_mask(image)
            class_idx = int(np.argmax(np.array(classes) == row["type_cell"]))
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": bytes_feature(image.tobytes()),
                        "mask": bytes_feature(mask.tobytes()),
                        "class": int64_feature(class_idx),
                    }
                )
            )
            writer.write(example.SerializeToString())
            written += 1
    return written


def parse_tfrecord(feature: tf.Tensor, img_shape: Tuple[int, int] = IMG_SHAPE):
    features = tf.io.parse_single_example(
        feature,
        features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    image = tf.reshape(
        tf.io.decode_raw(features["image"], out_type=tf.uint8),
        shape=(img_shape[0], img_shape[1], 3),
    )
    mask = tf.reshape(
        tf.io.decode_raw(features["mask"], out_type=tf.uint8),
        shape=(img_shape[0], img_shape[1], 1),
    )
    class_idx = tf.reshape(tf.cast(features["class"], tf.int64), (1,))
    return image, mask, class_idx


def save_processed_split(dataset: tf.data.Dataset, split_name: str, class_names: Sequence[str], processed_dir: Path) -> Dict[str, int]:
    counters = {class_name: 0 for class_name in class_names}
    for image, mask, class_idx in dataset:
        image_np = image.numpy().astype(np.uint8)
        mask_np = mask.numpy().astype(np.uint8)
        cls = int(tf.squeeze(class_idx).numpy())
        class_name = class_names[cls]

        img_dir = processed_dir / split_name / "images" / class_name
        mask_dir = processed_dir / split_name / "masks" / class_name
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        idx = counters[class_name]
        image_path = img_dir / f"img_{idx:04d}.png"
        mask_path = mask_dir / f"mask_{idx:04d}.png"

        cv2.imwrite(str(image_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask_np.squeeze())
        counters[class_name] += 1
    return counters


def build_raw_dataset_from_tfrecord(path: Path, img_shape: Tuple[int, int] = IMG_SHAPE) -> tf.data.Dataset:
    return tf.data.TFRecordDataset(str(path)).map(
        lambda feature: parse_tfrecord(feature, img_shape=img_shape),
        num_parallel_calls=AUTOTUNE,
    )


def classification_task(image: tf.Tensor, mask: tf.Tensor, class_idx: tf.Tensor, num_classes: int):
    del mask
    class_ohe = tf.one_hot(indices=tf.squeeze(class_idx, axis=0), depth=num_classes)
    return tf.cast(image, dtype=tf.float32) / 255.0, tf.cast(class_ohe, dtype=tf.float32)


def notebook_best_training_task(
    image: tf.Tensor,
    mask: tf.Tensor,
    class_idx: tf.Tensor,
    num_classes: int,
    seed: int = SEED,
):
    del mask
    augmented = tf.image.random_flip_left_right(image, seed=seed)
    augmented = tf.image.random_flip_up_down(augmented, seed=seed)
    augmented = tf.image.random_brightness(augmented, max_delta=0.01, seed=seed)
    augmented = tf.image.random_contrast(augmented, lower=0.5, upper=1.0, seed=seed)
    augmented = tf.image.random_saturation(augmented, lower=0.5, upper=1.0, seed=seed)
    del augmented
    class_ohe = tf.one_hot(indices=tf.squeeze(class_idx, axis=0), depth=num_classes)

    # El notebook calcula una imagen aumentada, pero termina devolviendo la original.
    return tf.cast(image, dtype=tf.float32) / 255.0, tf.cast(class_ohe, dtype=tf.float32)


def build_best_classification_datasets(
    tfrecord_paths: SplitPaths,
    class_names: Sequence[str],
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> Dict[str, tf.data.Dataset]:
    train_raw = build_raw_dataset_from_tfrecord(tfrecord_paths.train)
    val_raw = build_raw_dataset_from_tfrecord(tfrecord_paths.val)
    test_raw = build_raw_dataset_from_tfrecord(tfrecord_paths.test)

    train = train_raw.map(
        lambda image, mask, class_idx: notebook_best_training_task(image, mask, class_idx, len(class_names), seed),
        num_parallel_calls=AUTOTUNE,
    )
    val = val_raw.map(
        lambda image, mask, class_idx: classification_task(image, mask, class_idx, len(class_names)),
        num_parallel_calls=AUTOTUNE,
    )
    test = test_raw.map(
        lambda image, mask, class_idx: classification_task(image, mask, class_idx, len(class_names)),
        num_parallel_calls=AUTOTUNE,
    )

    return {
        "train": train.batch(batch_size).prefetch(AUTOTUNE),
        "val": val.batch(batch_size).prefetch(AUTOTUNE),
        "test": test.batch(batch_size).prefetch(AUTOTUNE),
    }


def write_metadata(
    output_dir: Path,
    raw_dir: Path,
    classes: Sequence[str],
    split_frames: Dict[str, pd.DataFrame],
    tfrecord_counts: Dict[str, int],
    img_shape: Tuple[int, int],
    seed: int,
) -> None:
    metadata = {
        "raw_dir": str(raw_dir.resolve()),
        "classes": list(classes),
        "img_shape": list(img_shape),
        "seed": seed,
        "split_sizes": {split: len(frame) for split, frame in split_frames.items()},
        "tfrecord_counts": tfrecord_counts,
        "best_classification_model_in_notebook": {
            "name": "Model with VGG16",
            "training_input": "train_data_clf_aug",
            "validation_input": "val_data_clf",
            "test_input": "test_data_clf",
            "precision": 0.506236,
            "auc_roc": 0.927239,
            "accuracy": 0.318275,
            "note": "El notebook crea una imagen aumentada pero retorna la original en el pipeline ganador.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_data(
    raw_dir: Path,
    output_dir: Path,
    img_shape: Tuple[int, int],
    seed: int,
    export_processed_png: bool,
) -> Dict[str, object]:
    set_seed(seed)
    classes = discover_classes(raw_dir)
    df_imgs = build_image_dataframe(raw_dir, classes)
    split_frames = split_dataframe(df_imgs, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for split_name, frame in split_frames.items():
        frame.to_csv(metadata_dir / f"{split_name}.csv", index=False)

    tfrecords_dir = output_dir / "tfrecords"
    tfrecord_paths = SplitPaths(
        train=tfrecords_dir / "blood_cell_cancer_with_mask_train.tfrecord",
        val=tfrecords_dir / "blood_cell_cancer_with_mask_val.tfrecord",
        test=tfrecords_dir / "blood_cell_cancer_with_mask_test.tfrecord",
    )

    tfrecord_counts = {
        "train": create_tfrecord(tfrecord_paths.train, split_frames["train"], classes, img_shape),
        "val": create_tfrecord(tfrecord_paths.val, split_frames["val"], classes, img_shape),
        "test": create_tfrecord(tfrecord_paths.test, split_frames["test"], classes, img_shape),
    }

    processed_counts = None
    if export_processed_png:
        processed_dir = output_dir / "processed_data"
        processed_counts = {}
        processed_counts["train"] = save_processed_split(
            build_raw_dataset_from_tfrecord(tfrecord_paths.train, img_shape),
            "train",
            classes,
            processed_dir,
        )
        processed_counts["val"] = save_processed_split(
            build_raw_dataset_from_tfrecord(tfrecord_paths.val, img_shape),
            "val",
            classes,
            processed_dir,
        )
        processed_counts["test"] = save_processed_split(
            build_raw_dataset_from_tfrecord(tfrecord_paths.test, img_shape),
            "test",
            classes,
            processed_dir,
        )

    write_metadata(output_dir, raw_dir, classes, split_frames, tfrecord_counts, img_shape, seed)

    return {
        "classes": classes,
        "splits": {name: len(frame) for name, frame in split_frames.items()},
        "tfrecord_paths": {key: str(value.resolve()) for key, value in asdict(tfrecord_paths).items()},
        "tfrecord_counts": tfrecord_counts,
        "processed_counts": processed_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara el dataset de Blood Cell Cancer desde datos raw hasta el punto de entrada del entrenamiento ganador."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("Blood cell Cancer [ALL]"),
        help="Directorio con las carpetas de clases crudas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "blood_cell_cancer_prepared",
        help="Directorio donde se guardaran metadata, TFRecords y PNG procesados.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=IMG_SHAPE[0],
        help="Alto de las imagenes redimensionadas.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=IMG_SHAPE[1],
        help="Ancho de las imagenes redimensionadas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla para splits y operaciones reproducibles.",
    )
    parser.add_argument(
        "--skip-processed-png",
        action="store_true",
        help="Evita exportar la copia procesada en PNG y solo genera TFRecords y metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_shape = (args.height, args.width)
    result = prepare_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        img_shape=img_shape,
        seed=args.seed,
        export_processed_png=not args.skip_processed_png,
    )
    print(json.dumps(result, indent=2))
    print("\nPara reconstruir el dataset de entrenamiento ganador, importa build_best_classification_datasets().")


if __name__ == "__main__":
    main()
