#!/bin/bash
#SBATCH --job-name=grpo_train_short      # Job name for the GRPO training
#SBATCH --partition=gpu,gpu-preempt      # Try 'gpu' first, then 'gpu-preempt'
#SBATCH --qos=short                      # Use 'short' QOS for higher priority (max 4h)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                # 8 CPUs are good for data loading and evaluation threads
#SBATCH --mem=80G                        # Ample RAM for the 4B model and dataset
#SBATCH --time=03:59:00                  # Max time for 'short' QOS
#SBATCH --output=logs/grpo_train_short_%j.out # Log output to a 'logs' directory
#SBATCH --error=logs/grpo_train_short_%j.err  # Log errors to a 'logs' directory
#SBATCH --gpus=1
#SBATCH --constraint="a100"         # Optimized for modern GPUs which Unsloth/VLLM prefer


# Job information
echo "==============================================="
echo "GRPO Clinical Note Training Job"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "SLURM_GPUS: ${SLURM_GPUS:-not set}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-not set}"
echo "SLURM_GPU_BIND: ${SLURM_GPU_BIND:-not set}"
echo "==============================================="

# Create logs directory if it doesn't exist
mkdir -p logs grpo_training/{models,checkpoints,outputs,lora}

# Load required modules (adjust based on your cluster's module system)
echo "Loading required modules..."
module purge 2>/dev/null || true

# Try to load CUDA modules (common names on different clusters)
CUDA_LOADED=false
CUDA_MODULES=("cuda/12.1" "cuda/11.8" "cuda" "CUDA/12.1" "CUDA/11.8" "nvidia/cuda/12.1" "nvidia/cuda/11.8")

for cuda_module in "${CUDA_MODULES[@]}"; do
    if module load "$cuda_module" 2>/dev/null; then
        echo "✅ Loaded CUDA module: $cuda_module"
        CUDA_LOADED=true
        break
    fi
done

if [ "$CUDA_LOADED" = false ]; then
    echo "⚠️  Warning: Could not load CUDA module. Trying without explicit module loading."
    echo "Available modules containing 'cuda':"
    module avail cuda 2>&1 | head -10 || echo "Module system not available or no cuda modules found"
fi

# Try to load GCC if available (some clusters need compatible GCC for CUDA)
GCC_MODULES=("gcc/9.3.0" "gcc/11.2.0" "gcc/12.1.0" "gcc" "GCC/11.2.0")
for gcc_module in "${GCC_MODULES[@]}"; do
    if module load "$gcc_module" 2>/dev/null; then
        echo "✅ Loaded GCC module: $gcc_module"
        break
    fi
done

# Initialize conda using the system installation
echo "Initializing conda..."
CONDA_BASE="/modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Found conda at: $CONDA_BASE"
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    # Add your conda envs directory to conda's search path
    export CONDA_ENVS_PATH="/work/pi_jaimedavila_umass_edu/dwinkelman_umass_edu/.conda/envs:$CONDA_ENVS_PATH"

    echo "Conda environments search path: $CONDA_ENVS_PATH"
    echo "Available environments on compute node:"
    conda env list

    echo "Attempting to activate conda environment: grpo-training"

    # Try activating by name first
    if conda activate grpo-training 2>/dev/null; then
        echo "✅ Successfully activated environment: grpo-training"
    else
        echo "Name-based activation failed, trying path-based activation..."

        # Try activating by full path
        CONDA_ENV_PATH="/work/pi_jaimedavila_umass_edu/dwinkelman_umass_edu/.conda/envs/grpo-training"
        if conda activate "$CONDA_ENV_PATH" 2>/dev/null; then
            echo "✅ Successfully activated environment using path: $CONDA_ENV_PATH"
        else
            echo "Conda activation failed, trying direct environment activation..."

            # Fallback: direct activation without conda command
            if [ -f "$CONDA_ENV_PATH/bin/activate" ]; then
                source "$CONDA_ENV_PATH/bin/activate"
                export PATH="$CONDA_ENV_PATH/bin:$PATH"
                export CONDA_DEFAULT_ENV="grpo-training"
                export CONDA_PREFIX="$CONDA_ENV_PATH"
                echo "✅ Activated environment using direct activation"
            else
                echo "ERROR: All activation methods failed"
                echo "Checking if environment directory exists:"
                ls -la "$CONDA_ENV_PATH"
                exit 1
            fi
        fi
    fi

else
    echo "ERROR: Could not find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

# Verify environment
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check GPU availability and CUDA setup
echo "==============================================="
echo "GPU AND CUDA DIAGNOSTICS"
echo "==============================================="

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi found, checking GPU status..."
    nvidia-smi
    echo ""

    # Check CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    echo "SLURM_GPUS: ${SLURM_GPUS:-not set}"
    echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-not set}"

    # Check GPU count
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Number of GPUs detected: $GPU_COUNT"

    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "ERROR: No GPUs detected by nvidia-smi"
        echo "This may be a SLURM allocation issue"
        exit 1
    fi
else
    echo "ERROR: nvidia-smi not found"
    echo "NVIDIA drivers may not be installed or loaded"
    exit 1
fi

# Test CUDA with Python
echo "Testing CUDA availability with Python..."
python -c "
import sys
print(f'Python executable: {sys.executable}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
    else:
        print('CUDA not available to PyTorch')
        print('Checking CUDA compilation...')
        print(f'PyTorch built with CUDA: {torch.version.cuda is not None}')
        sys.exit(1)
except ImportError as e:
    print(f'Error importing torch: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not properly available to PyTorch"
    exit 1
fi

# Test if we can import unsloth without errors
echo "Testing Unsloth import..."
python -c "
try:
    from unsloth import FastLanguageModel
    print('✅ Unsloth imported successfully')
except Exception as e:
    print(f'❌ Error importing Unsloth: {e}')
    import sys
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to import Unsloth"
    exit 1
fi

echo "✅ GPU and CUDA checks passed"
echo "==============================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Additional CUDA environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Try to find CUDA installation if not set
if [ ! -d "$CUDA_HOME" ]; then
    CUDA_PATHS=("/usr/local/cuda" "/opt/cuda" "/usr/cuda" "/software/cuda" "/apps/cuda")
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if [ -d "$cuda_path" ]; then
            export CUDA_HOME="$cuda_path"
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            echo "Found CUDA at: $CUDA_HOME"
            break
        fi
    done
fi

echo "CUDA_HOME: ${CUDA_HOME}"
echo "PATH: ${PATH}"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# Verify required environment variables
required_vars=("HF_API_KEY" "WANDB_API_KEY" "GEMINI_API_KEY" "OPENROUTER_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "ERROR: Missing required environment variables: ${missing_vars[*]}"
    echo "Please set these variables in your .env file before submitting the job."
    exit 1
fi

echo "All required environment variables are set."
echo "==============================================="

# Run the training script
echo "Starting GRPO training..."
python trainer.py \

# Capture exit code
exit_code=$?

echo "==============================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
fi
echo "==============================================="

exit $exit_code
