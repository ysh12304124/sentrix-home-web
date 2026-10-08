#!/usr/bin/env bash
# ── 平台参数层 ────────────────────────────────────────────────────────────────
#
# 三台机器（153 主机 / 118 AGX Orin / 46 Orin NX）跑**同一份代码**，差异分两层：
#
#   1. 能探测的   → backend/platform_profile.py（CUDA 是否真能用、ffmpeg 有哪些
#                   硬件解码器、内存多大、装了 qdrant 没有）。**不需要任何参数**。
#   2. 探测不出的 → 本文件。有些差异是「这台机器上我们选择这么做」，不是能力问题。
#                   最典型的是 153 的 CLIP_DEVICE=cpu —— CUDA 明明可用（实跑自检
#                   通过），但 153 的 GPU 驱动与 NVML 不匹配会让 CUDA caching
#                   allocator 崩，所以这里**主动**选 CPU。探测永远猜不到这一条。
#
# 用法：由 start_sentrix_api.sh 在读完 .env 之后 source。
#   - 默认按 /proc/device-tree/model 自动识别平台
#   - SENTRIX_PLATFORM=<host|orin-agx|orin-nx|jetson> 可强制覆盖（排障用）
#   - 这里只写**覆盖值**：已经设置过的变量一律不覆盖（`:=` 而非 `=`），
#     所以 .env 与命令行仍然有最终决定权
#   - 每一项都必须写清「为什么」，否则下一个人不敢改

sentrix_detect_platform() {
  if [[ -n "${SENTRIX_PLATFORM:-}" ]]; then
    echo "$SENTRIX_PLATFORM"
    return
  fi
  if [[ -r /proc/device-tree/model ]]; then
    local model
    model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
    case "$model" in
      *"Orin NX"*)  echo "orin-nx";  return ;;
      *"AGX Orin"*) echo "orin-agx"; return ;;
      *[Jj]etson*)  echo "jetson";   return ;;
    esac
  fi
  echo "host"
}

sentrix_apply_platform_profile() {
  local platform
  platform="$(sentrix_detect_platform)"
  export SENTRIX_PLATFORM="$platform"

  case "$platform" in
    host)
      # 153：CUDA 自检能过，但驱动/NVML 不匹配会让 CUDA caching allocator 崩，
      # 视觉编码必须走 CPU。（118 的提交 15227b8 记着「原 cpu 是照搬 153 的驱动问题」
      # —— 反过来说明这条只对 153 成立，Orin 上不该照抄。）
      export CLIP_DEVICE="${CLIP_DEVICE:-cpu}"
      # 153 是 x86 + 独立显存，AdaFace 的 700MB fp32 权重放得下，且低质量人脸更稳。
      export FACE_EMBEDDING_MODE="${FACE_EMBEDDING_MODE:-adaface}"
      ;;

    orin-agx|orin-nx|jetson)
      # Orin 是统一内存：CLIP 走 CUDA 才有实用速度（实测 CPU 21~36 秒/张，
      # CUDA 0.23~0.82 秒/张）。
      export CLIP_DEVICE="${CLIP_DEVICE:-cuda}"
      # Orin 上默认 buffalo_l（174MB ONNX）而不是 AdaFace（700MB fp32 PyTorch）：
      # 16GB 统一内存里这 700MB 是压垮内存的那一根稻草，且 AdaFace 在 153 上本身
      # 就跑 CPU。face_embeddings 的契约禁止跨模型静默回退，所以这里必须显式选。
      export FACE_EMBEDDING_MODE="${FACE_EMBEDDING_MODE:-legacy}"
      ;;

    *)
      echo "警告：未知平台 '$platform'，使用代码探测的默认值" >&2
      ;;
  esac

  # 人脸/检索权重路径：按 $HOME 推导，不写死用户名。
  # 153 的 retinaface 在 ~/benchmarks/retinaface/，AdaFace 在仓库内 models/AdaFace/。
  # 显式设置过就尊重（.env / 命令行优先）。
  export RETINAFACE_MODEL_PATH="${RETINAFACE_MODEL_PATH:-$HOME/benchmarks/retinaface/retinaface_r50.onnx}"
  if [[ -d "$root/models/AdaFace" ]]; then
    export ADAFACE_MODEL_PATH="${ADAFACE_MODEL_PATH:-$root/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt}"
    export ADAFACE_REPO_ROOT="${ADAFACE_REPO_ROOT:-$root/models/AdaFace}"
  fi

  # 并发不在这里写死：由 platform_profile.pipeline_workers() 按内存推算
  # （16GB→2、48GB+→8），需要压过时用 .env 的 SENTRIX_PIPELINE_MAX_WORKERS。
  # 向量后端同理，默认已是 qdrant；仅当机器没装 qdrant-client 才回落 sqlite。
}
