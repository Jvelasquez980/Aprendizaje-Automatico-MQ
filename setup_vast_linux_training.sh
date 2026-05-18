#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="gpu"
SKIP_VERIFY=0

usage() {
  cat <<'EOF'
Uso:
  bash setup_vast_linux_training.sh [--gpu|--cpu] [--venv-dir PATH] [--python BIN] [--skip-verify]

Opciones:
  --gpu            Configura el entorno para entrenamiento con GPU en Vast.ai.
  --cpu            Configura el entorno solo para CPU.
  --venv-dir PATH  Ruta alternativa para el entorno virtual.
  --python BIN     Binario de Python a usar. Por defecto: python3
  --skip-verify    Omite la verificacion final de imports y versiones.
  -h, --help       Muestra esta ayuda.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      MODE="gpu"
      shift
      ;;
    --cpu)
      MODE="cpu"
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Opcion no reconocida: $1" >&2
      usage
      exit 1
      ;;
  esac
done

log() {
  echo
  echo "[setup] $1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Comando requerido no encontrado: $1" >&2
    exit 1
  fi
}

log "Verificando herramientas base"
require_command "$PYTHON_BIN"
require_command git

if [[ "$MODE" == "gpu" ]]; then
  require_command nvidia-smi
  log "GPU detectada"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

log "Creando entorno virtual en ${VENV_DIR}"
"$PYTHON_BIN" -m venv "$VENV_DIR"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "No se pudo encontrar el script de activacion del entorno virtual." >&2
  exit 1
fi

log "Actualizando pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

log "Instalando dependencias del proyecto"
python -m pip install -r "${ROOT_DIR}/requierements.txt"

if [[ "$MODE" == "gpu" ]]; then
  log "Reemplazando qiskit-aer por la variante con soporte GPU"
  python -m pip uninstall -y qiskit-aer || true
  python -m pip install "qiskit-aer-gpu==0.17.2"
fi

log "Creando directorios de artefactos"
mkdir -p \
  "${ROOT_DIR}/artifacts" \
  "${ROOT_DIR}/artifacts/blood_cell_cancer_prepared" \
  "${ROOT_DIR}/artifacts/blood_cell_cancer_training" \
  "${ROOT_DIR}/artifacts/blood_cell_cancer_quantum"

if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  log "Verificando imports y backend de Qiskit Aer"
  python <<'PY'
import json
import sys

checks = {}

try:
    import numpy
    checks["numpy"] = numpy.__version__
except Exception as exc:
    checks["numpy"] = f"ERROR: {exc}"

try:
    import pandas
    checks["pandas"] = pandas.__version__
except Exception as exc:
    checks["pandas"] = f"ERROR: {exc}"

try:
    import tensorflow as tf
    checks["tensorflow"] = tf.__version__
except Exception as exc:
    checks["tensorflow"] = f"ERROR: {exc}"

try:
    import qiskit
    checks["qiskit"] = qiskit.__version__
except Exception as exc:
    checks["qiskit"] = f"ERROR: {exc}"

try:
    import qiskit_machine_learning
    checks["qiskit_machine_learning"] = qiskit_machine_learning.__version__
except Exception as exc:
    checks["qiskit_machine_learning"] = f"ERROR: {exc}"

try:
    import qiskit_algorithms
    checks["qiskit_algorithms"] = qiskit_algorithms.__version__
except Exception as exc:
    checks["qiskit_algorithms"] = f"ERROR: {exc}"

try:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    checks["qiskit_aer"] = {
        "devices": list(backend.available_devices()),
        "methods": list(backend.available_methods()),
    }
except Exception as exc:
    checks["qiskit_aer"] = f"ERROR: {exc}"

print(json.dumps(checks, indent=2))

errors = [value for value in checks.values() if isinstance(value, str) and value.startswith("ERROR:")]
if errors:
    sys.exit(1)
PY
fi

log "Setup finalizado"
cat <<EOF

Entorno virtual:
  source "${VENV_DIR}/bin/activate"

Ejemplos de uso posteriores:
  python prepare_blood_cell_cancer_data.py --raw-dir "Blood cell Cancer [ALL]"
  python encode_blood_cell_cancer_amplitude.py --prepared-dir artifacts/blood_cell_cancer_prepared
  python train_blood_cell_cancer_vqc_aer_gpu.py --data-dir artifacts/blood_cell_cancer_quantum/amplitude_encoding --representation masks --num-qubits 16 --device $( [[ "$MODE" == "gpu" ]] && echo GPU || echo CPU ) --precision single

EOF
