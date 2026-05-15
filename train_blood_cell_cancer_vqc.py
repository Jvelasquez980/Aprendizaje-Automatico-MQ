import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library.raw_feature_vector import raw_feature_vector
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_DATA_DIR = Path("artifacts") / "blood_cell_cancer_quantum" / "amplitude_encoding"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "blood_cell_cancer_quantum" / "vqc"
DEFAULT_REPRESENTATION = "masks"
DEFAULT_SEED = 42


@dataclass
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    paths: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def split_npz_path(data_dir: Path, split: str, representation: str) -> Path:
    path = data_dir / f"amplitude_{representation}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo esperado: {path}")
    return path


def load_split(data_dir: Path, split: str, representation: str) -> Tuple[DatasetSplit, np.ndarray]:
    data = np.load(split_npz_path(data_dir, split, representation), allow_pickle=True)
    dataset = DatasetSplit(
        X=data["X"].astype(np.float64),
        y=data["y"].astype(np.int64),
        paths=data["paths"],
    )
    classes = data["classes"]
    return dataset, classes


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = np.zeros_like(X, dtype=np.float64)
    non_zero = norms.squeeze(axis=1) > 0
    normalized[non_zero] = X[non_zero] / norms[non_zero]
    normalized[~non_zero, 0] = 1.0
    return normalized


def maybe_reduce_dimension(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    target_qubits: int,
    output_dir: Path,
    seed: int,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, Dict[str, object]]:
    target_dim = 2 ** target_qubits
    input_dim = train.X.shape[1]
    reducer_info: Dict[str, object] = {
        "input_dimension": int(input_dim),
        "target_dimension": int(target_dim),
        "method": "none",
    }

    if input_dim == target_dim:
        return train, val, test, reducer_info

    if input_dim < target_dim:
        raise ValueError(
            f"La dimension de entrada ({input_dim}) es menor que la requerida por {target_qubits} qubits ({target_dim})."
        )

    if train.X.shape[0] >= target_dim:
        reducer = PCA(n_components=target_dim, svd_solver="randomized", random_state=seed)
        X_train = reducer.fit_transform(train.X)
        X_val = reducer.transform(val.X)
        X_test = reducer.transform(test.X)

        reducer_path = output_dir / f"pca_reducer_{target_qubits}q.joblib"
        joblib.dump(reducer, reducer_path)

        reducer_info = {
            "input_dimension": int(input_dim),
            "target_dimension": int(target_dim),
            "method": "pca",
            "explained_variance_ratio_sum": float(np.sum(reducer.explained_variance_ratio_)),
            "reducer_path": str(reducer_path.resolve()),
        }
    else:
        variances = np.var(train.X, axis=0)
        selected_idx = np.argsort(variances)[::-1][:target_dim]
        X_train = train.X[:, selected_idx]
        X_val = val.X[:, selected_idx]
        X_test = test.X[:, selected_idx]

        reducer_path = output_dir / f"variance_selector_{target_qubits}q.joblib"
        joblib.dump({"selected_idx": selected_idx}, reducer_path)

        reducer_info = {
            "input_dimension": int(input_dim),
            "target_dimension": int(target_dim),
            "method": "variance_topk",
            "reducer_path": str(reducer_path.resolve()),
        }

    reduced_train = DatasetSplit(X=l2_normalize_rows(X_train), y=train.y, paths=train.paths)
    reduced_val = DatasetSplit(X=l2_normalize_rows(X_val), y=val.y, paths=val.paths)
    reduced_test = DatasetSplit(X=l2_normalize_rows(X_test), y=test.y, paths=test.paths)
    return reduced_train, reduced_val, reduced_test, reducer_info


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_history_csv(history: List[Dict[str, float]], history_path: Path) -> None:
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "objective_value"])
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str,
    weights: np.ndarray,
    iteration: int,
    objective_value: float,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.npz"
    np.savez_compressed(
        checkpoint_path,
        weights=np.asarray(weights, dtype=np.float64),
        iteration=np.int64(iteration),
        objective_value=np.float64(objective_value),
    )
    return checkpoint_path


def load_resume_weights(resume_path: Path) -> np.ndarray:
    if not resume_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint solicitado para reanudar: {resume_path}")
    data = np.load(resume_path, allow_pickle=True)
    return np.asarray(data["weights"], dtype=np.float64)


def build_optimizer(name: str, maxiter: int, seed: int, num_qubits: int, reps: int):
    num_parameters = 2 * num_qubits * reps
    if name.lower() == "cobyla":
        effective_maxiter = max(maxiter, num_parameters + 2)
        return COBYLA(maxiter=effective_maxiter)
    if name.lower() == "spsa":
        return SPSA(maxiter=maxiter, learning_rate=0.05, perturbation=0.05, last_avg=1, resamplings=1)
    raise ValueError(f"Optimizador no soportado: {name}")


def build_vqc(
    num_qubits: int,
    reps: int,
    optimizer_name: str,
    maxiter: int,
    callback_store: list,
    seed: int,
    initial_point: np.ndarray | None = None,
    warm_start: bool = False,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 10,
    history_path: Path | None = None,
    verbose: bool = True,
) -> VQC:
    if verbose:
        print(f"Construyendo feature_map raw_feature_vector para {2 ** num_qubits} amplitudes...", flush=True)
    feature_map = raw_feature_vector(2 ** num_qubits)
    if verbose:
        print(f"Construyendo ansatz real_amplitudes para {num_qubits} qubits con reps={reps}...", flush=True)
    ansatz = real_amplitudes(num_qubits, reps=reps)
    best_objective = {"value": float("inf")}

    def callback(weights: np.ndarray, objective_value: float) -> None:
        iteration = len(callback_store) + 1
        callback_store.append(
            {
                "iteration": iteration,
                "objective_value": float(objective_value),
            }
        )
        if verbose:
            print(f"[Iter {iteration}] objective={float(objective_value):.8f}", flush=True)
        if history_path is not None:
            save_history_csv(callback_store, history_path)
        if checkpoint_dir is not None and checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(checkpoint_dir, "latest_checkpoint", weights, iteration, float(objective_value))
            if verbose:
                print(f"Checkpoint guardado: {checkpoint_path}", flush=True)
        if checkpoint_dir is not None and objective_value < best_objective["value"]:
            best_objective["value"] = float(objective_value)
            best_path = save_checkpoint(checkpoint_dir, "best_checkpoint", weights, iteration, float(objective_value))
            if verbose:
                print(f"Nuevo mejor checkpoint: {best_path}", flush=True)

    if verbose:
        print(f"Inicializando VQC con optimizer={optimizer_name} y maxiter={maxiter}...", flush=True)
    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=build_optimizer(optimizer_name, maxiter, seed, num_qubits, reps),
        sampler=StatevectorSampler(seed=seed),
        interpret=lambda x: x % 4,
        output_shape=4,
        callback=callback,
        initial_point=initial_point,
        warm_start=warm_start,
    )


def evaluate_split(model: VQC, split: DatasetSplit) -> Dict[str, float]:
    y_pred = np.asarray(model.predict(split.X)).astype(np.int64)
    return compute_metrics(split.y, y_pred)


def train_vqc(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    classes: np.ndarray,
    output_dir: Path,
    num_qubits: int,
    reps: int,
    optimizer_name: str,
    maxiter: int,
    seed: int,
    reducer_info: Dict[str, object],
    checkpoint_interval: int,
    resume_from: Path | None,
    verbose: bool,
) -> Dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    callback_history = []
    history_path = output_dir / f"vqc_{num_qubits}q_history.csv"
    checkpoint_dir = output_dir / "checkpoints"
    if verbose:
        print(f"Preparando salida en: {output_dir}", flush=True)
    initial_point = load_resume_weights(resume_from) if resume_from is not None else None
    model = build_vqc(
        num_qubits,
        reps,
        optimizer_name,
        maxiter,
        callback_history,
        seed,
        initial_point=initial_point,
        warm_start=initial_point is not None,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        history_path=history_path,
        verbose=verbose,
    )
    if verbose:
        print(
            f"Iniciando entrenamiento VQC: qubits={num_qubits}, reps={reps}, optimizer={optimizer_name}, maxiter={maxiter}",
            flush=True,
        )
        if initial_point is not None:
            print(f"Reanudando desde checkpoint: {resume_from}", flush=True)
        print(f"Llamando model.fit() con {train.X.shape[0]} muestras y dimension {train.X.shape[1]}...", flush=True)
    model.fit(train.X, train.y)
    if verbose:
        print("Entrenamiento finalizado. Calculando metricas...", flush=True)

    train_metrics = evaluate_split(model, train)
    val_metrics = evaluate_split(model, val)
    test_metrics = evaluate_split(model, test)

    weights = getattr(model, "weights", None)
    if weights is None:
        weights = getattr(model, "_fit_result", None)

    weights_path = output_dir / f"vqc_{num_qubits}q_weights.npy"
    np.save(weights_path, np.asarray(model.weights, dtype=np.float64))

    save_history_csv(callback_history, history_path)
    latest_checkpoint_path = save_checkpoint(
        checkpoint_dir,
        "latest_checkpoint",
        np.asarray(model.weights, dtype=np.float64),
        len(callback_history),
        float(callback_history[-1]["objective_value"]) if callback_history else float("nan"),
    )

    config = {
        "num_qubits": num_qubits,
        "feature_dimension": 2 ** num_qubits,
        "state_initialization": "Amplitude encoding con raw_feature_vector de qiskit_machine_learning.",
        "ansatz": f"real_amplitudes(num_qubits={num_qubits}, reps={reps})",
        "measurement": "Interpretacion multiclase con modulo 4 sobre el bitstring medido.",
        "loss": "cross_entropy",
        "optimizer": optimizer_name.lower(),
        "optimizer_maxiter": maxiter,
        "sampler": "StatevectorSampler",
        "reducer_info": reducer_info,
        "classes": [str(c) for c in classes],
        "weights_path": str(weights_path.resolve()),
        "history_path": str(history_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_interval": checkpoint_interval,
        "resume_from": str(resume_from.resolve()) if resume_from is not None else None,
        "latest_checkpoint_path": str(latest_checkpoint_path.resolve()),
        "best_checkpoint_path": str((checkpoint_dir / "best_checkpoint.npz").resolve()),
    }
    (output_dir / f"vqc_{num_qubits}q_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = {
        "config": config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "score_train": float(model.score(train.X, train.y)),
        "score_val": float(model.score(val.X, val.y)),
        "score_test": float(model.score(test.X, test.y)),
    }
    (output_dir / f"vqc_{num_qubits}q_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena un Variational Quantum Classifier con Qiskit sobre los datos amplitude-encoded de blood cell cancer."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directorio con los .npz de amplitude encoding.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directorio para pesos, historial y metricas.")
    parser.add_argument(
        "--representation",
        choices=["masks", "images"],
        default=DEFAULT_REPRESENTATION,
        help="Representacion a consumir dentro de los .npz generados.",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        choices=[8, 16],
        default=8,
        help="Numero de qubits del VQC. Con 8 qubits se reduce a 256 dimensiones antes del entrenamiento.",
    )
    parser.add_argument("--reps", type=int, default=2, help="Numero de repeticiones del ansatz real_amplitudes.")
    parser.add_argument(
        "--optimizer",
        choices=["cobyla", "spsa"],
        default="cobyla",
        help="Optimizador para el VQC.",
    )
    parser.add_argument("--maxiter", type=int, default=50, help="Iteraciones maximas del optimizador.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla de reproducibilidad.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Guardar un checkpoint ligero cada N iteraciones del optimizador.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Ruta a un checkpoint .npz previamente guardado para reanudar desde esos pesos.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Desactiva los mensajes de progreso en consola.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.output_dir / f"{args.representation}_{args.num_qubits}q"
    run_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet
    if verbose:
        print(f"Cargando datos desde: {args.data_dir}", flush=True)
        print(f"Representacion seleccionada: {args.representation}", flush=True)
        print(f"Configuracion solicitada: num_qubits={args.num_qubits}, reps={args.reps}, optimizer={args.optimizer}, maxiter={args.maxiter}", flush=True)

    train, classes = load_split(args.data_dir, "train", args.representation)
    val, _ = load_split(args.data_dir, "val", args.representation)
    test, _ = load_split(args.data_dir, "test", args.representation)
    if verbose:
        print(
            f"Splits cargados: train={train.X.shape}, val={val.X.shape}, test={test.X.shape}",
            flush=True,
        )

    if verbose:
        print("Aplicando ajuste de dimension si es necesario...", flush=True)
    train, val, test, reducer_info = maybe_reduce_dimension(
        train,
        val,
        test,
        args.num_qubits,
        run_dir,
        args.seed,
    )
    if verbose:
        print(
            f"Dimension final tras preprocesamiento: train={train.X.shape}, val={val.X.shape}, test={test.X.shape}",
            flush=True,
        )
        print(f"Reducer info: {json.dumps(reducer_info, indent=2)}", flush=True)

    results = train_vqc(
        train=train,
        val=val,
        test=test,
        classes=classes,
        output_dir=run_dir,
        num_qubits=args.num_qubits,
        reps=args.reps,
        optimizer_name=args.optimizer,
        maxiter=args.maxiter,
        seed=args.seed,
        reducer_info=reducer_info,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        verbose=verbose,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
