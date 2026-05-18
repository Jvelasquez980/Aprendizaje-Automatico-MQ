import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import real_amplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.circuit.library.raw_feature_vector import raw_feature_vector
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_DATA_DIR = Path("artifacts") / "blood_cell_cancer_quantum" / "amplitude_encoding"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "blood_cell_cancer_quantum" / "vqc_aer_gpu"
DEFAULT_REPRESENTATION = "masks"
DEFAULT_SEED = 42


@dataclass
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    paths: np.ndarray


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


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
    return dataset, data["classes"]


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = np.zeros_like(X, dtype=np.float64)
    non_zero = norms.squeeze(axis=1) > 0
    normalized[non_zero] = X[non_zero] / norms[non_zero]
    normalized[~non_zero, 0] = 1.0
    return normalized


def canonicalize_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    vec = np.asarray(amplitudes, dtype=np.float64).copy()
    if not np.all(np.isfinite(vec)):
        raise ValueError("Se encontraron amplitudes no finitas.")
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        vec[:] = 0.0
        vec[0] = 1.0
        return vec
    vec /= norm
    if vec.size > 1:
        tail_power = float(np.sum(vec[1:] * vec[1:]))
        head_sq = max(0.0, 1.0 - tail_power)
        vec[0] = np.copysign(np.sqrt(head_sq), vec[0] if vec[0] != 0 else 1.0)
    else:
        vec[0] = 1.0
    final_norm = np.linalg.norm(vec)
    if final_norm == 0 or not np.isfinite(final_norm):
        vec[:] = 0.0
        vec[0] = 1.0
    else:
        vec /= final_norm
    return vec


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


def save_history_csv(history: list[Dict[str, float]], history_path: Path) -> None:
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "nfev", "objective_value", "step_size"])
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(checkpoint_dir: Path, checkpoint_name: str, weights: np.ndarray, iteration: int, objective_value: float) -> Path:
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
    data = np.load(resume_path, allow_pickle=True)
    return np.asarray(data["weights"], dtype=np.float64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def clip_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


class AerAmplitudeVQC:
    def __init__(
        self,
        num_qubits: int,
        reps: int,
        shots: int,
        device: str,
        precision: str,
        enable_custatevec: bool,
        verbose: bool = True,
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.shots = shots
        self.device = device.upper()
        self.precision = precision
        self.enable_custatevec = enable_custatevec
        self.verbose = verbose

        if verbose:
            print(f"Construyendo feature_map raw_feature_vector para {2 ** num_qubits} amplitudes...", flush=True)
        self.feature_map = raw_feature_vector(2 ** num_qubits)
        if verbose:
            print(f"Construyendo ansatz real_amplitudes para {num_qubits} qubits con reps={reps}...", flush=True)
        self.ansatz = real_amplitudes(num_qubits, reps=reps, entanglement="linear")
        self.circuit = self.feature_map.compose(self.ansatz)
        self.circuit.measure_all()
        self.feature_params = list(self.feature_map.parameters)
        self.weight_params = list(self.ansatz.parameters)
        self.num_parameters = len(self.weight_params)
        if verbose:
            print(f"Inicializando AerSimulator con device={self.device}, precision={self.precision}...", flush=True)
        self.backend = self._build_backend()

    def _build_backend(self) -> AerSimulator:
        backend_options = {
            "method": "statevector",
            "device": self.device,
            "precision": self.precision,
        }
        if self.enable_custatevec:
            backend_options["cuStateVec_enable"] = True
        backend = AerSimulator(**backend_options)
        available_devices = tuple(backend.available_devices())
        if self.device not in available_devices:
            raise RuntimeError(
                f"El backend Aer disponible en este entorno no soporta device={self.device}. Dispositivos: {available_devices}"
            )
        return backend

    def build_bound_circuit(self, amplitudes: np.ndarray, weights: np.ndarray):
        safe_amplitudes = canonicalize_amplitudes(amplitudes)
        assignments = {}
        for parameter, value in zip(self.feature_params, safe_amplitudes):
            assignments[parameter] = float(value)
        for parameter, value in zip(self.weight_params, weights):
            assignments[parameter] = float(value)
        bound = self.circuit.assign_parameters(assignments)
        return bound.decompose(reps=1)

    def bitstring_to_class(self, bitstring: str) -> int:
        return int(bitstring, 2) % 4

    def counts_to_probabilities(self, counts: Dict[str, int]) -> np.ndarray:
        probs = np.zeros(4, dtype=np.float64)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            probs[self.bitstring_to_class(bitstring)] += count / total
        return clip_probabilities(probs.reshape(1, -1))[0]

    def predict_proba(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        circuits = [self.build_bound_circuit(amplitudes, weights) for amplitudes in X]
        transpiled = transpile(circuits, self.backend, optimization_level=0)
        result = self.backend.run(transpiled, shots=self.shots).result()
        outputs = np.zeros((len(circuits), 4), dtype=np.float64)
        for idx in range(len(circuits)):
            counts = result.get_counts(idx)
            outputs[idx] = self.counts_to_probabilities(counts)
        return outputs

    def predict(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X, weights), axis=1)


def stratified_subset(split: DatasetSplit, per_class_limit: int | None) -> DatasetSplit:
    if per_class_limit is None:
        return split
    indices = []
    for label in np.unique(split.y):
        class_idx = np.where(split.y == label)[0][:per_class_limit]
        indices.extend(class_idx.tolist())
    indices = np.array(sorted(indices), dtype=np.int64)
    return DatasetSplit(X=split.X[indices], y=split.y[indices], paths=split.paths[indices])


def evaluate_split(model: AerAmplitudeVQC, split: DatasetSplit, weights: np.ndarray, eval_limit_per_class: int | None) -> Dict[str, float]:
    subset = stratified_subset(split, eval_limit_per_class)
    y_pred = model.predict(subset.X, weights)
    metrics = compute_metrics(subset.y, y_pred)
    metrics["evaluated_samples"] = int(len(subset.y))
    return metrics


def train_aer_vqc(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    classes: np.ndarray,
    output_dir: Path,
    num_qubits: int,
    reps: int,
    maxiter: int,
    batch_size: int,
    shots: int,
    learning_rate: float,
    perturbation: float,
    seed: int,
    reducer_info: Dict[str, object],
    checkpoint_interval: int,
    resume_from: Path | None,
    eval_limit_per_class: int | None,
    device: str,
    precision: str,
    enable_custatevec: bool,
    verbose: bool,
) -> Dict[str, object]:
    rng = set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    history_path = output_dir / f"aer_vqc_{num_qubits}q_history.csv"

    model = AerAmplitudeVQC(
        num_qubits=num_qubits,
        reps=reps,
        shots=shots,
        device=device,
        precision=precision,
        enable_custatevec=enable_custatevec,
        verbose=verbose,
    )
    if verbose:
        print(f"Numero de parametros entrenables: {model.num_parameters}", flush=True)

    if resume_from is not None:
        weights0 = load_resume_weights(resume_from)
        if verbose:
            print(f"Reanudando desde checkpoint: {resume_from}", flush=True)
    else:
        weights0 = rng.normal(loc=0.0, scale=0.1, size=model.num_parameters)

    history: list[Dict[str, float]] = []
    best_objective = {"value": float("inf")}

    def sample_batch() -> tuple[np.ndarray, np.ndarray]:
        size = min(batch_size, len(train.y))
        indices = rng.choice(len(train.y), size=size, replace=False)
        return train.X[indices], train.y[indices]

    def objective(weights: np.ndarray) -> float:
        X_batch, y_batch = sample_batch()
        probs = model.predict_proba(X_batch, weights)
        losses = -np.log(probs[np.arange(len(y_batch)), y_batch])
        return float(np.mean(losses))

    def callback(nfev: int, weights: np.ndarray, objective_value: float, step_size: float, accepted: bool) -> None:
        del accepted
        iteration = len(history) + 1
        history.append(
            {
                "iteration": iteration,
                "nfev": int(nfev),
                "objective_value": float(objective_value),
                "step_size": float(step_size),
            }
        )
        if verbose:
            print(f"[Iter {iteration}] nfev={nfev} objective={float(objective_value):.8f} step={float(step_size):.8f}", flush=True)
        save_history_csv(history, history_path)
        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            latest_path = save_checkpoint(checkpoint_dir, "latest_checkpoint", weights, iteration, float(objective_value))
            if verbose:
                print(f"Checkpoint guardado: {latest_path}", flush=True)
        if objective_value < best_objective["value"]:
            best_objective["value"] = float(objective_value)
            best_path = save_checkpoint(checkpoint_dir, "best_checkpoint", weights, iteration, float(objective_value))
            if verbose:
                print(f"Nuevo mejor checkpoint: {best_path}", flush=True)

    optimizer = SPSA(
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        callback=callback,
        last_avg=1,
        resamplings=1,
    )

    if verbose:
        print(
            f"Iniciando entrenamiento Aer VQC: qubits={num_qubits}, reps={reps}, batch_size={batch_size}, shots={shots}, maxiter={maxiter}",
            flush=True,
        )
    start = time.time()
    result = optimizer.minimize(fun=objective, x0=weights0)
    duration = time.time() - start
    final_weights = np.asarray(result.x, dtype=np.float64)

    weights_path = output_dir / f"aer_vqc_{num_qubits}q_weights.npy"
    np.save(weights_path, final_weights)
    latest_checkpoint_path = save_checkpoint(
        checkpoint_dir,
        "latest_checkpoint",
        final_weights,
        len(history),
        float(history[-1]["objective_value"]) if history else float("nan"),
    )

    if verbose:
        print("Entrenamiento finalizado. Calculando metricas...", flush=True)
    train_metrics = evaluate_split(model, train, final_weights, eval_limit_per_class)
    val_metrics = evaluate_split(model, val, final_weights, eval_limit_per_class)
    test_metrics = evaluate_split(model, test, final_weights, eval_limit_per_class)

    config = {
        "num_qubits": num_qubits,
        "feature_dimension": 2 ** num_qubits,
        "state_initialization": "Amplitude encoding con raw_feature_vector, ejecutado en AerSimulator.",
        "ansatz": f"real_amplitudes(num_qubits={num_qubits}, reps={reps}, entanglement='linear')",
        "measurement": "Medicion de todos los qubits y mapeo de bitstring a clase mediante modulo 4.",
        "optimizer": "SPSA manual",
        "backend": {
            "device": device.upper(),
            "precision": precision,
            "enable_custatevec": enable_custatevec,
            "shots": shots,
        },
        "maxiter": maxiter,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "perturbation": perturbation,
        "reducer_info": reducer_info,
        "classes": [str(c) for c in classes],
        "weights_path": str(weights_path.resolve()),
        "history_path": str(history_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_interval": checkpoint_interval,
        "resume_from": str(resume_from.resolve()) if resume_from is not None else None,
        "latest_checkpoint_path": str(latest_checkpoint_path.resolve()),
        "best_checkpoint_path": str((checkpoint_dir / "best_checkpoint.npz").resolve()),
        "eval_limit_per_class": eval_limit_per_class,
        "duration_seconds": duration,
    }
    (output_dir / f"aer_vqc_{num_qubits}q_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = {
        "config": config,
        "optimizer_fun": float(result.fun),
        "optimizer_nfev": int(result.nfev),
        "optimizer_nit": int(result.nit),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / f"aer_vqc_{num_qubits}q_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena un flujo cuántico manual usando Qiskit Aer, orientado a aprovechar GPU."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directorio con los .npz de amplitude encoding.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directorio para pesos, historial y metricas.")
    parser.add_argument("--representation", choices=["masks", "images"], default=DEFAULT_REPRESENTATION, help="Representacion a consumir dentro de los .npz.")
    parser.add_argument("--num-qubits", type=int, choices=[8, 16], default=16, help="Numero de qubits del flujo Aer.")
    parser.add_argument("--reps", type=int, default=1, help="Numero de repeticiones del ansatz.")
    parser.add_argument("--maxiter", type=int, default=50, help="Iteraciones maximas de SPSA.")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamano de minibatch para la funcion objetivo.")
    parser.add_argument("--shots", type=int, default=512, help="Numero de shots por evaluacion de circuito.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate de SPSA.")
    parser.add_argument("--perturbation", type=float, default=0.05, help="Perturbacion de SPSA.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla de reproducibilidad.")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Guardar checkpoint cada N iteraciones.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint .npz para reanudar pesos.")
    parser.add_argument("--eval-limit-per-class", type=int, default=32, help="Limite estratificado por clase para evaluar mas rapido. Usa 0 para evaluar todo.")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU", help="Dispositivo objetivo para AerSimulator.")
    parser.add_argument("--precision", choices=["single", "double"], default="single", help="Precision del simulador Aer.")
    parser.add_argument("--enable-custatevec", action="store_true", help="Intenta habilitar cuStateVec si el entorno lo soporta.")
    parser.add_argument("--quiet", action="store_true", help="Desactiva mensajes de progreso en consola.")
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    run_dir = args.output_dir / f"{args.representation}_{args.num_qubits}q"
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Cargando datos desde: {args.data_dir}", flush=True)
        print(f"Representacion seleccionada: {args.representation}", flush=True)
        print(
            f"Configuracion solicitada: num_qubits={args.num_qubits}, reps={args.reps}, device={args.device}, precision={args.precision}, maxiter={args.maxiter}",
            flush=True,
        )

    train, classes = load_split(args.data_dir, "train", args.representation)
    val, _ = load_split(args.data_dir, "val", args.representation)
    test, _ = load_split(args.data_dir, "test", args.representation)
    if verbose:
        print(f"Splits cargados: train={train.X.shape}, val={val.X.shape}, test={test.X.shape}", flush=True)
        print("Aplicando ajuste de dimension si es necesario...", flush=True)

    train, val, test, reducer_info = maybe_reduce_dimension(train, val, test, args.num_qubits, run_dir, args.seed)
    if verbose:
        print(f"Dimension final: train={train.X.shape}, val={val.X.shape}, test={test.X.shape}", flush=True)
        print(f"Reducer info: {json.dumps(reducer_info, indent=2)}", flush=True)

    eval_limit = None if args.eval_limit_per_class == 0 else args.eval_limit_per_class
    results = train_aer_vqc(
        train=train,
        val=val,
        test=test,
        classes=classes,
        output_dir=run_dir,
        num_qubits=args.num_qubits,
        reps=args.reps,
        maxiter=args.maxiter,
        batch_size=args.batch_size,
        shots=args.shots,
        learning_rate=args.learning_rate,
        perturbation=args.perturbation,
        seed=args.seed,
        reducer_info=reducer_info,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        eval_limit_per_class=eval_limit,
        device=args.device,
        precision=args.precision,
        enable_custatevec=args.enable_custatevec,
        verbose=verbose,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
