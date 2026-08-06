# 12B Liveness Report

{
  "model": "gemma4:12b",
  "endpoint": "http://127.0.0.1:11434",
  "checks": {
    "tags": {
      "ok": true,
      "models": [
        "gemma4:12b"
      ]
    },
    "model_present": {
      "ok": true,
      "expected": "gemma4:12b",
      "actual": [
        "gemma4:12b"
      ]
    },
    "chat_roundtrip": {
      "ok": true,
      "status": 200,
      "actual_model": "gemma4:12b",
      "latency_s": 3.14,
      "sample": ""
    },
    "nvidia_smi": {
      "ok": true,
      "output": "name, driver_version, memory.total [MiB], memory.used [MiB], utilization.gpu [%]\nNVIDIA GeForce RTX 3090, 595.84, 24576 MiB, 11343 MiB, 94 %"
    },
    "ollama_residency": {
      "ok": true,
      "resident": [
        {
          "name": "gemma4:12b",
          "size_vram": 8424301526,
          "expires_at": "2026-08-06T12:49:48.341461388+08:00"
        }
      ]
    }
  },
  "verdict": "ALIVE",
  "stop_condition": "proceed to V2"
}