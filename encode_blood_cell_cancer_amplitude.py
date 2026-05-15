import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_PREPARED_DIR = Path("artifacts") / "blood_cell_cancer_prepared"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "blood_cell_cancer_quantum" / "amplitude_encoding"
DEFAULT_SPLITS = ("train", "val", "test")


def load_metadata(prepared_dir: Path) -> Dict[str, object]:
    metadata_path = prepared_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontro metadata.json en {prepared_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def get_processed_data_dir(prepared_dir: Path) -> Path:
    processed_dir = prepared_dir / "processed_data"
    if not processed_dir.exists():
        raise FileNotFoundError(f"No se encontro el directorio de datos procesados: {processed_dir}")
    return processed_dir


def list_image_paths(
    processed_data_dir: Path,
    split: str,
    class_names: Sequence[str],
    use_masks: bool,
) -> List[Tuple[Path, int]]:
    subdir = "masks" if use_masks else "images"
    base = processed_data_dir / split / subdir
    if not base.exists():
        raise FileNotFoundError(f"No existe el split solicitado: {base}")

    samples: List[Tuple[Path, int]] = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = base / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                samples.append((img_path, label_idx))
    return samples


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << int(np.ceil(np.log2(value)))


def infer_reference_size(metadata: Dict[str, object], processed_data_dir: Path, split: str, use_masks: bool) -> Tuple[int, int]:
    img_shape = metadata.get("img_shape")
    if isinstance(img_shape, list) and len(img_shape) == 2:
        return int(img_shape[0]), int(img_shape[1])

    samples = list_image_paths(processed_data_dir, split, metadata["classes"], use_masks)
    if not samples:
        raise ValueError("No se pudo inferir el tamano de referencia porque no hay muestras disponibles.")
    with Image.open(samples[0][0]) as image:
        width, height = image.size
    return height, width


def image_to_amplitude_vector(img_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    image = Image.open(img_path).convert("L")
    image = image.resize((img_size[1], img_size[0]))

    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()

    vec = arr.flatten()
    target_len = next_power_of_two(len(vec))
    pad_len = target_len - len(vec)
    if pad_len > 0:
        vec = np.pad(vec, (0, pad_len), mode="constant", constant_values=0.0)

    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.zeros_like(vec)
        vec[0] = 1.0
    else:
        vec = vec / norm
    return vec.astype(np.float32)


def encode_split_to_amplitudes(
    processed_data_dir: Path,
    split: str,
    class_names: Sequence[str],
    use_masks: bool,
    img_size: Tuple[int, int],
    max_samples_per_class: int | None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    samples = list_image_paths(processed_data_dir, split, class_names, use_masks)

    if max_samples_per_class is not None:
        counts = {idx: 0 for idx in range(len(class_names))}
        filtered = []
        for path, label in samples:
            if counts[label] < max_samples_per_class:
                filtered.append((path, label))
                counts[label] += 1
        samples = filtered

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []

    for img_path, label in samples:
        vectors.append(image_to_amplitude_vector(img_path, img_size=img_size))
        labels.append(label)
        paths.append(str(img_path.resolve()))

    if not vectors:
        padded_len = next_power_of_two(img_size[0] * img_size[1])
        return np.empty((0, padded_len), dtype=np.float32), np.array([], dtype=np.int64), []

    X = np.stack(vectors, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, paths


def save_split_npz(
    out_dir: Path,
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    paths: Sequence[str],
    img_size: Tuple[int, int],
    class_names: Sequence[str],
    representation: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"amplitude_{representation}_{split}.npz"
    np.savez_compressed(
        save_path,
        X=X,
        y=y,
        paths=np.array(paths, dtype=object),
        img_size=np.array(img_size, dtype=np.int64),
        classes=np.array(class_names, dtype=object),
        representation=representation,
        encoded_dim=np.int64(X.shape[1] if X.ndim == 2 and X.shape[0] > 0 else next_power_of_two(img_size[0] * img_size[1])),
    )
    return save_path


def encode_all_splits_and_save(
    prepared_dir: Path,
    out_dir: Path,
    use_masks: bool,
    img_size: Tuple[int, int] | None,
    max_samples_per_class: int | None,
) -> Dict[str, object]:
    metadata = load_metadata(prepared_dir)
    processed_data_dir = get_processed_data_dir(prepared_dir)
    class_names = metadata["classes"]
    if img_size is None:
        img_size = infer_reference_size(metadata, processed_data_dir, "train", use_masks)

    representation = "masks" if use_masks else "images"
    saved_files: Dict[str, str] = {}
    split_shapes: Dict[str, Tuple[int, int]] = {}

    for split in DEFAULT_SPLITS:
        X, y, paths = encode_split_to_amplitudes(
            processed_data_dir=processed_data_dir,
            split=split,
            class_names=class_names,
            use_masks=use_masks,
            img_size=img_size,
            max_samples_per_class=max_samples_per_class,
        )
        save_path = save_split_npz(
            out_dir=out_dir,
            split=split,
            X=X,
            y=y,
            paths=paths,
            img_size=img_size,
            class_names=class_names,
            representation=representation,
        )
        saved_files[split] = str(save_path.resolve())
        split_shapes[split] = tuple(X.shape)

    summary = {
        "prepared_dir": str(prepared_dir.resolve()),
        "processed_data_dir": str(processed_data_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "representation": representation,
        "img_size": list(img_size),
        "encoded_dimension": next_power_of_two(img_size[0] * img_size[1]),
        "classes": class_names,
        "max_samples_per_class": max_samples_per_class,
        "saved_files": saved_files,
        "split_shapes": {key: list(value) for key, value in split_shapes.items()},
    }
    (out_dir / f"amplitude_{representation}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def amplitude_vector_to_image(amplitudes: np.ndarray, img_size: Tuple[int, int]) -> Image.Image:
    expected_len = img_size[0] * img_size[1]
    if len(amplitudes) < expected_len:
        raise ValueError("La longitud del vector es menor que el tamano esperado de imagen.")

    vec = np.abs(amplitudes.astype(np.float32)[:expected_len])
    if vec.max() > 0:
        vec = vec / vec.max()

    arr = vec.reshape((img_size[0], img_size[1]))
    arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="L")


def save_decoded_preview(npz_path: Path, out_dir: Path, num_samples: int) -> None:
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = data["classes"]
    img_size = tuple(int(v) for v in data["img_size"])

    out_dir.mkdir(parents=True, exist_ok=True)
    total = min(num_samples, X.shape[0])
    for index in range(total):
        image = amplitude_vector_to_image(X[index], img_size)
        label = str(classes[int(y[index])])
        image.save(out_dir / f"decoded_{index:04d}_class-{label}.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convierte los datos preparados de blood cell cancer a vectores listos para amplitude encoding."
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
        help="Directorio generado por prepare_blood_cell_cancer_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde se guardaran los .npz codificados.",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Usa imagenes RGB convertidas a escala de grises en vez de mascaras.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Alto para redimensionar antes del amplitude encoding.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ancho para redimensionar antes del amplitude encoding.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Limita la cantidad de muestras por clase en cada split.",
    )
    parser.add_argument(
        "--decoded-preview-split",
        choices=list(DEFAULT_SPLITS),
        default=None,
        help="Si se indica, genera una pequena muestra decodificada desde el .npz resultante.",
    )
    parser.add_argument(
        "--decoded-preview-count",
        type=int,
        default=8,
        help="Cantidad de muestras a reconstruir para el preview decodificado.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = None
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Debes proporcionar ambos argumentos: --height y --width.")
        img_size = (args.height, args.width)

    summary = encode_all_splits_and_save(
        prepared_dir=args.prepared_dir,
        out_dir=args.output_dir,
        use_masks=not args.use_images,
        img_size=img_size,
        max_samples_per_class=args.max_samples_per_class,
    )

    if args.decoded_preview_split is not None:
        representation = summary["representation"]
        npz_path = Path(summary["saved_files"][args.decoded_preview_split])
        preview_dir = Path(summary["output_dir"]) / f"decoded_preview_{representation}_{args.decoded_preview_split}"
        save_decoded_preview(npz_path, preview_dir, args.decoded_preview_count)
        summary["decoded_preview_dir"] = str(preview_dir.resolve())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
