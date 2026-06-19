#!/bin/bash
# Generates per-model v2 run scripts + HTCondor submit files in /fast/jtaraz/cluster_stuff.
set -euo pipefail
cd /fast/jtaraz/cluster_stuff
mkdir -p test_173

REPO=/fast/jtaraz/LIARS/lie-detector-for-LLMs
VENV=/fast/jtaraz/lie-detector-for-LLMs/.venv

# gen <label> <model_id> <tp> <gpus> <cuda_module 0/1> <extra_args> <allow_a100 0/1>
gen() {
  local label="$1" model="$2" tp="$3" gpus="$4" cuda="$5" extra="$6" a100="$7"
  local sh="run_v2_${label}.sh" sub="job_v2_${label}.sub"

  local cudablock=""
  if [ "$cuda" = "1" ]; then
    cudablock=$'source /etc/profile.d/modules.sh\nmodule load cuda/12.9'
  fi

  cat > "$sh" <<EOF
#!/bin/bash
export HOME="\${HOME:-/home/jtaraz}"
set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="\$HOME/.local/bin:\$PATH"

set -a; source /fast/jtaraz/cluster_stuff/.env; set +a
export HF_HOME=/fast/jtaraz/hf_cache
mkdir -p "\$HF_HOME"
${cudablock}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0

VENV=${VENV}
REPO=${REPO}
source "\$VENV/bin/activate"

echo "=== v2 generate: ${model} n=100 (TP=${tp}) ==="
python "\$REPO/insider-trading/run_v2.py" \\
    --models ${model} \\
    --n 100 \\
    --tensor-parallel-size ${tp} \\
    ${extra} \\
    --output-dir "\$REPO/insider-trading/data/v2"
echo "=== done: ${label} ==="
EOF
  chmod +x "$sh"

  # resource sizing by gpu count
  local mem cpus disk
  case "$gpus" in
    1) mem=65536;  cpus=4;  disk=100G ;;
    2) mem=131072; cpus=4;  disk=150G ;;
    4) mem=262144; cpus=8;  disk=200G ;;
    8) mem=262144; cpus=16; disk=200G ;;
    *) mem=131072; cpus=4;  disk=150G ;;
  esac

  local req='requirements = TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"'
  if [ "$a100" = "1" ]; then
    req='requirements = (TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")'
  fi

  cat > "$sub" <<EOF
executable = /bin/bash
arguments = ${sh}
environment = "OPENAI_API_KEY=\$ENV(OPENAI_API_KEY) HF_TOKEN=\$ENV(HF_TOKEN)"
error = test_173/v2_${label}_\$(Cluster).err
output = test_173/v2_${label}_\$(Cluster).out
log = test_173/v2_${label}_\$(Cluster).log
request_memory = ${mem}
request_cpus = ${cpus}
request_gpus = ${gpus}
${req}
request_disk = ${disk}
+BypassLXCfs="true"
queue
EOF
  echo "generated $sh + $sub  (model=$model tp=$tp gpus=$gpus)"
}

gen llama31_8b   meta-llama/Llama-3.1-8B-Instruct          1 1 0 "--max-model-len 16384" 0
gen gemma2_9b    google/gemma-2-9b-it                      1 1 0 "--max-model-len 16384" 0
gen gemma3_12b   google/gemma-3-12b-it                     1 1 0 "--max-model-len 16384" 0
gen qwen25_32b   Qwen/Qwen2.5-32B-Instruct                 2 2 0 "--max-model-len 16384" 0
gen gemma3_27b   google/gemma-3-27b-it                     2 2 0 "--max-model-len 16384" 0
gen qwq_32b      Qwen/QwQ-32B                              2 2 0 "--max-model-len 32768 --max-tokens 8192" 0
gen r1distill_32b deepseek-ai/DeepSeek-R1-Distill-Qwen-32B 2 2 0 "--max-model-len 32768 --max-tokens 8192" 0
gen qwen25_72b   Qwen/Qwen2.5-72B-Instruct                 4 4 0 "--max-model-len 16384" 0
gen llama33_70b  meta-llama/Llama-3.3-70B-Instruct         4 4 0 "--max-model-len 16384" 0
gen gptoss_120b  openai/gpt-oss-120b                       2 2 1 "--max-model-len 16384" 0
gen kimi_k2      QuixiAI/Kimi-K2-Instruct-AWQ              8 8 1 "--gpu-memory-utilization 0.97 --max-model-len 8192 --enforce-eager" 1

echo "ALL GENERATED"
