#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY = Path(os.environ.get(
    "SENTRIX_VLLM_REGISTRY",
    str(_REPO_ROOT / "configs/sentrix_vllm_registry_192_168_0_153.json")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "profile"


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def service_dirs(registry: dict[str, Any]) -> dict[str, Path]:
    root = Path(registry["service_root"])
    dirs = {
        "root": root,
        "run": root / "run",
        "log": root / "log",
        "cmd": root / "cmd",
        "state": root / "state",
    }
    for item in dirs.values():
        item.mkdir(parents=True, exist_ok=True)
    return dirs


def service_paths(registry: dict[str, Any], profile_name: str) -> dict[str, Path]:
    dirs = service_dirs(registry)
    stem = safe_name(profile_name)
    return {
        "pid": dirs["run"] / f"{stem}.pid",
        "log": dirs["log"] / f"{stem}.log",
        "cmd": dirs["cmd"] / f"{stem}.json",
    }


def state_path(registry: dict[str, Any]) -> Path:
    if registry.get("state_file"):
        path = Path(registry["state_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return service_dirs(registry)["state"] / "current.json"


def lock_path(registry: dict[str, Any]) -> Path:
    return service_dirs(registry)["run"] / "manager.lock"


@contextmanager
def manager_lock(registry: dict[str, Any]):
    path = lock_path(registry)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return None


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def pid_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def process_cmdline(pid: int) -> str:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        return proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def is_vllm_process(pid: int) -> bool:
    cmdline = process_cmdline(pid)
    return (not cmdline) or ("vllm" in cmdline and "serve" in cmdline)


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def resolve_profile(registry: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(registry["profiles"][name])
    except KeyError:
        choices = ", ".join(sorted(registry.get("profiles", {})))
        raise SystemExit(f"unknown profile: {name}\navailable profiles: {choices}")


def parse_lora_module(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("LoRA module must be NAME=PATH")
    name, path = raw.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("LoRA module must be NAME=PATH")
    return {"name": name, "path": path}


def apply_runtime_overrides(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(profile)
    scalar_overrides = {
        "model": "model",
        "host": "host",
        "port": "port",
        "served_model_name": "served_model_name",
        "dtype": "dtype",
        "quantization": "quantization",
        "quantization_config": "quantization_config",
        "load_format": "load_format",
        "max_model_len": "max_model_len",
        "max_num_seqs": "max_num_seqs",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "gpu_memory_utilization": "gpu_memory_utilization",
        "tensor_parallel_size": "tensor_parallel_size",
        "kv_cache_dtype": "kv_cache_dtype",
        "max_lora_rank": "max_lora_rank",
        "lora_dtype": "lora_dtype",
        "default_max_tokens": "default_max_tokens",
        "cuda_visible_devices": "cuda_visible_devices",
    }
    for arg_name, key in scalar_overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            updated[key] = value

    for arg_name, key in (
        ("enable_lora", "enable_lora"),
        ("fully_sharded_loras", "fully_sharded_loras"),
        ("enable_prefix_caching", "enable_prefix_caching"),
        ("enable_chunked_prefill", "enable_chunked_prefill"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            updated[key] = bool(value)

    if getattr(args, "lora_module", None):
        updated["lora_modules"] = args.lora_module
        updated["enable_lora"] = True

    if getattr(args, "extra_arg", None):
        updated["extra_args"] = list(updated.get("extra_args", [])) + list(args.extra_arg)

    return updated


def selected_host_port(registry: dict[str, Any], profile: dict[str, Any]) -> tuple[str, int]:
    host = str(profile.get("host") or registry.get("default_host", "127.0.0.1"))
    port = int(profile.get("port") or registry.get("default_port", 8100))
    return host, port


def build_command(
    registry: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
) -> tuple[list[str], dict[str, str], int]:
    selected_host, selected_port = selected_host_port(registry, profile)
    vllm_binary = registry["vllm_binary"]
    command = [
        vllm_binary,
        "serve",
        profile["model"],
        "--host",
        selected_host,
        "--port",
        str(selected_port),
    ]

    served_name = profile.get("served_model_name", profile_name)
    command.append("--served-model-name")
    if isinstance(served_name, list):
        command += [str(item) for item in served_name]
    else:
        command.append(str(served_name))

    simple_flags = {
        "dtype": "--dtype",
        "quantization": "--quantization",
        "quantization_config": "--quantization-config",
        "load_format": "--load-format",
        "tensor_parallel_size": "--tensor-parallel-size",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_model_len": "--max-model-len",
        "max_num_seqs": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "kv_cache_dtype": "--kv-cache-dtype",
    }
    for key, flag in simple_flags.items():
        if key in profile and profile[key] is not None:
            command += [flag, str(profile[key])]

    if profile.get("trust_remote_code", False):
        command.append("--trust-remote-code")
    if "mm_processor_cache_gb" in profile:
        command += ["--mm-processor-cache-gb", str(profile["mm_processor_cache_gb"])]
    if "limit_mm_per_prompt" in profile:
        command += [
            "--limit-mm-per-prompt",
            json.dumps(profile["limit_mm_per_prompt"], separators=(",", ":")),
        ]
    if "enable_prefix_caching" in profile:
        command.append("--enable-prefix-caching" if profile["enable_prefix_caching"] else "--no-enable-prefix-caching")
    if "enable_chunked_prefill" in profile:
        command.append("--enable-chunked-prefill" if profile["enable_chunked_prefill"] else "--no-enable-chunked-prefill")

    if profile.get("enable_lora", False):
        command.append("--enable-lora")
        command += ["--max-lora-rank", str(profile.get("max_lora_rank", 64))]
        if profile.get("lora_dtype"):
            command += ["--lora-dtype", str(profile["lora_dtype"])]
        if "fully_sharded_loras" in profile:
            command.append("--fully-sharded-loras" if profile["fully_sharded_loras"] else "--no-fully-sharded-loras")
        modules = profile.get("lora_modules", [])
        if modules:
            command.append("--lora-modules")
            for module in modules:
                command.append(f"{module['name']}={module['path']}")

    command += [str(item) for item in profile.get("extra_args", [])]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if profile.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(profile["cuda_visible_devices"])

    cuda_home = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
    if cuda_home.exists():
        env["CUDA_HOME"] = str(cuda_home)
        cuda_bin = cuda_home / "bin"
        cuda_lib64 = cuda_home / "lib64"
        cuda_targets_lib64 = cuda_home / "targets" / "x86_64-linux" / "lib"
        cuda_targets_include = cuda_home / "targets" / "x86_64-linux" / "include"
        env["PATH"] = ":".join([str(cuda_bin), env.get("PATH", "")])
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(cuda_lib64), str(cuda_targets_lib64), env.get("LD_LIBRARY_PATH", "")]
        )
        env["CPATH"] = ":".join([str(cuda_targets_include), env.get("CPATH", "")])

    if profile.get("allow_runtime_lora", False):
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    return command, env, selected_port


def validate_paths(profile: dict[str, Any]) -> int:
    model_path = Path(profile["model"])
    if model_path.is_absolute() and not model_path.exists():
        print(f"model path is missing: {model_path}", file=sys.stderr)
        return 2
    for module in profile.get("lora_modules", []):
        path = Path(module["path"])
        if not path.exists():
            print(f"LoRA path is missing: {path}", file=sys.stderr)
            return 2
    return 0


def read_current_state(registry: dict[str, Any]) -> dict[str, Any] | None:
    return read_json(state_path(registry))


def state_active(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    try:
        pid = int(state.get("pid"))
    except Exception:
        return False
    return pid_active(pid)


def clear_current_state(registry: dict[str, Any], state: dict[str, Any] | None = None) -> None:
    remove_file(state_path(registry))
    if state and state.get("profile"):
        remove_file(service_paths(registry, str(state["profile"]))["pid"])


def stop_state(registry: dict[str, Any], state: dict[str, Any], timeout: int, kill: bool, force: bool) -> int:
    try:
        pid = int(state["pid"])
    except Exception:
        clear_current_state(registry, state)
        print("cleared invalid current state")
        return 0

    if not pid_active(pid):
        clear_current_state(registry, state)
        print(f"cleared stale current state pid={pid}")
        return 0

    if not is_vllm_process(pid) and not force:
        cmdline = process_cmdline(pid)
        print(f"refuse to stop non-vLLM-looking pid={pid}: {cmdline}", file=sys.stderr)
        print("use --force only if you have verified this pid belongs to the managed vLLM instance", file=sys.stderr)
        return 3

    target_pgid = int(state.get("pgid") or pid)
    try:
        os.killpg(target_pgid, signal.SIGTERM)
    except ProcessLookupError:
        clear_current_state(registry, state)
        print(f"cleared stale current state pid={pid}")
        return 0

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_active(pid):
            clear_current_state(registry, state)
            print(f"stopped profile={state.get('profile')} pid={pid}")
            return 0
        time.sleep(1)

    if kill:
        os.killpg(target_pgid, signal.SIGKILL)
        time.sleep(1)
        clear_current_state(registry, state)
        print(f"killed profile={state.get('profile')} pid={pid}")
        return 0

    print(f"still running profile={state.get('profile')} pid={pid}", file=sys.stderr)
    return 1


def http_models(port: int, timeout: float = 2.0) -> str:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as response:
        return response.read().decode("utf-8")


def tail_log(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-max_lines:])


def wait_until_ready(proc: subprocess.Popen[Any], port: int, log_path: Path, timeout: int) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ret = proc.poll()
        if ret is not None:
            print(f"vLLM exited early code={ret}", file=sys.stderr)
            log_tail = tail_log(log_path)
            if log_tail:
                print(log_tail, file=sys.stderr)
            return 1
        try:
            payload = http_models(port, timeout=2.0)
            print(payload[:1000])
            return 0
        except Exception:
            time.sleep(2)
    print(f"started but not ready after {timeout}s; check log={log_path}")
    return 0


def build_state(
    registry_path: Path,
    registry: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
    command: list[str],
    proc: subprocess.Popen[Any],
    log_path: Path,
    port: int,
) -> dict[str, Any]:
    host, _port = selected_host_port(registry, profile)
    served_name = profile.get("served_model_name", profile_name)
    return {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "profile": profile_name,
        "pid": proc.pid,
        "pgid": proc.pid,
        "host": host,
        "port": port,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "external_url_hint": f"http://192.168.0.153:{port}/v1" if host == "0.0.0.0" else None,
        "model": profile["model"],
        "served_model_name": served_name,
        "dtype": profile.get("dtype"),
        "quantization": profile.get("quantization"),
        "quantization_config": profile.get("quantization_config"),
        "load_format": profile.get("load_format"),
        "max_model_len": profile.get("max_model_len"),
        "max_num_seqs": profile.get("max_num_seqs"),
        "max_num_batched_tokens": profile.get("max_num_batched_tokens"),
        "gpu_memory_utilization": profile.get("gpu_memory_utilization"),
        "tensor_parallel_size": profile.get("tensor_parallel_size"),
        "kv_cache_dtype": profile.get("kv_cache_dtype"),
        "default_max_tokens": profile.get("default_max_tokens"),
        "enable_lora": profile.get("enable_lora", False),
        "lora_modules": profile.get("lora_modules", []),
        "cuda_visible_devices": profile.get("cuda_visible_devices"),
        "log_path": str(log_path),
        "registry_path": str(registry_path),
        "command": command,
        "notes": profile.get("notes"),
    }


def command_list(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    for name, raw_profile in registry["profiles"].items():
        profile = copy.deepcopy(raw_profile)
        model = Path(profile["model"])
        model_ok = (not model.is_absolute()) or model.exists()
        lora_state = []
        for module in profile.get("lora_modules", []):
            path = Path(module["path"])
            lora_state.append(f"{module['name']}:{'ok' if path.exists() else 'missing'}")
        print(
            f"{name}\tmodel={'ok' if model_ok else 'missing'}\t"
            f"served={profile.get('served_model_name', name)}\t"
            f"quant={profile.get('quantization') or '-'}\t"
            f"load={profile.get('load_format') or '-'}\t"
            f"ctx={profile.get('max_model_len', '-')}\t"
            f"seqs={profile.get('max_num_seqs', '-')}\t"
            f"gpu={profile.get('gpu_memory_utilization', '-')}\t"
            f"lora={','.join(lora_state) if lora_state else '-'}"
        )
        notes = profile.get("notes")
        if notes:
            print(f"  notes: {notes}")
    return 0


def command_cmd(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    profile = apply_runtime_overrides(resolve_profile(registry, args.profile), args)
    command, _env, port = build_command(registry, args.profile, profile)
    print(shlex.join(command))
    print(f"port={port}")
    if profile.get("default_max_tokens") is not None:
        print(f"default_max_tokens={profile['default_max_tokens']}  # marker/request-default only; OpenAI requests still pass max_tokens per call")
    return 0


def start_profile_locked(args: argparse.Namespace, *, switching: bool = False) -> int:
    registry = load_registry(args.registry)
    profile = apply_runtime_overrides(resolve_profile(registry, args.profile), args)
    paths = service_paths(registry, args.profile)

    current = read_current_state(registry)
    if args.dry_run:
        path_check = validate_paths(profile)
        command, _env, port = build_command(registry, args.profile, profile)
        if state_active(current):
            assert current is not None
            print(f"would_stop profile={current.get('profile')} pid={current.get('pid')}")
        elif current:
            print(f"would_clear_stale_state profile={current.get('profile')} pid={current.get('pid')}")
        print(shlex.join(command))
        print(f"dry_run=true port={port} log={paths['log']} state={state_path(registry)}")
        if profile.get("default_max_tokens") is not None:
            print(f"default_max_tokens={profile['default_max_tokens']}  # marker/request-default only; OpenAI requests still pass max_tokens per call")
        return path_check

    if state_active(current):
        assert current is not None
        same_profile = current.get("profile") == args.profile
        if not switching and not args.replace:
            if same_profile:
                print(f"{args.profile}: already running pid={current.get('pid')}")
                print(f"state={state_path(registry)}")
                return 0
            print(
                f"another managed profile is running: {current.get('profile')} pid={current.get('pid')}",
                file=sys.stderr,
            )
            print("use `switch <profile>` or `start <profile> --replace`", file=sys.stderr)
            return 3
        stop_result = stop_state(registry, current, timeout=args.timeout, kill=True, force=args.force)
        if stop_result != 0:
            return stop_result
    elif current:
        clear_current_state(registry, current)
        print(f"cleared stale current state profile={current.get('profile')} pid={current.get('pid')}")

    path_check = validate_paths(profile)
    if path_check:
        return path_check

    _host, port = selected_host_port(registry, profile)
    if port_open(port) and not args.force_port:
        print(f"port {port} is already open but not tracked by {state_path(registry)}", file=sys.stderr)
        print("refusing to overwrite an unmanaged service; use --force-port after manual verification", file=sys.stderr)
        return 4

    command, env, port = build_command(registry, args.profile, profile)

    paths["cmd"].write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["log"].open("ab") as log:
        log.write(f"\n\n[{utc_now()}] starting profile={args.profile}\n".encode("utf-8"))
        log.write((shlex.join(command) + "\n").encode("utf-8"))
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)

    paths["pid"].write_text(str(proc.pid), encoding="utf-8")
    state = build_state(args.registry, registry, args.profile, profile, command, proc, paths["log"], port)
    atomic_write_json(state_path(registry), state)
    print(f"started profile={args.profile} pid={proc.pid} port={port}")
    print(f"state={state_path(registry)}")
    print(f"log={paths['log']}")
    if args.wait_ready:
        return wait_until_ready(proc, port, paths["log"], args.ready_timeout)
    return 0


def command_start(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    with manager_lock(registry):
        return start_profile_locked(args, switching=False)


def command_switch(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    with manager_lock(registry):
        return start_profile_locked(args, switching=True)


def stop_profile_by_pid_file(registry: dict[str, Any], profile_name: str, timeout: int, kill: bool, force: bool) -> int:
    paths = service_paths(registry, profile_name)
    pid = read_pid(paths["pid"])
    if not pid or not pid_active(pid):
        remove_file(paths["pid"])
        print(f"{profile_name}: not running")
        return 0
    state = {
        "profile": profile_name,
        "pid": pid,
        "pgid": pid,
    }
    return stop_state(registry, state, timeout=timeout, kill=kill, force=force)


def command_stop(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    with manager_lock(registry):
        if args.profile:
            result = stop_profile_by_pid_file(registry, args.profile, args.timeout, args.kill, args.force)
            current = read_current_state(registry)
            if current and current.get("profile") == args.profile and not state_active(current):
                clear_current_state(registry, current)
            return result

        current = read_current_state(registry)
        if not current:
            print("no current managed vLLM instance")
            return 0
        return stop_state(registry, current, timeout=args.timeout, kill=args.kill, force=args.force)


def command_status(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    current = read_current_state(registry)
    if args.profile:
        paths = service_paths(registry, args.profile)
        pid = read_pid(paths["pid"])
        active = bool(pid and pid_active(pid))
        print(f"profile={args.profile} active={active} pid={pid or '-'}")
        print(f"log={paths['log']}")
        if active and args.check_http:
            profile = resolve_profile(registry, args.profile)
            _host, port = selected_host_port(registry, profile)
            try:
                print(http_models(port)[:1000])
            except Exception as exc:
                print(f"http_check_failed={exc}")
        return 0

    if not current:
        print("current=none")
        print(f"state={state_path(registry)}")
        return 0

    active = state_active(current)
    print(
        f"current={current.get('profile')} active={active} pid={current.get('pid')} "
        f"port={current.get('port')} served={current.get('served_model_name')}"
    )
    print(
        f"model={current.get('model')} quant={current.get('quantization') or '-'} "
        f"load={current.get('load_format') or '-'} ctx={current.get('max_model_len')} "
        f"seqs={current.get('max_num_seqs')} gpu={current.get('gpu_memory_utilization')}"
    )
    if current.get("default_max_tokens") is not None:
        print(f"default_max_tokens={current.get('default_max_tokens')}")
    print(f"state={state_path(registry)}")
    print(f"log={current.get('log_path')}")
    if active and args.check_http:
        try:
            print(http_models(int(current["port"]))[:1000])
        except Exception as exc:
            print(f"http_check_failed={exc}")
    return 0


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Override the model path or model id from the registry profile.")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--served-model-name")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float", "float16", "float32", "half"])
    parser.add_argument("--quantization")
    parser.add_argument("--quantization-config")
    parser.add_argument("--load-format")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int, help="vLLM request concurrency cap.")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--default-max-tokens", type=int, help="Recorded for benchmark clients; vLLM requests still pass max_tokens.")
    parser.add_argument("--cuda-visible-devices", help="Example: 0 or 0,1.")
    parser.add_argument("--enable-lora", dest="enable_lora", action="store_true", default=None)
    parser.add_argument("--disable-lora", dest="enable_lora", action="store_false")
    parser.add_argument("--lora-module", action="append", type=parse_lora_module, metavar="NAME=PATH")
    parser.add_argument("--max-lora-rank", type=int)
    parser.add_argument("--lora-dtype", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--fully-sharded-loras", dest="fully_sharded_loras", action="store_true", default=None)
    parser.add_argument("--no-fully-sharded-loras", dest="fully_sharded_loras", action="store_false")
    parser.add_argument("--enable-prefix-caching", dest="enable_prefix_caching", action="store_true", default=None)
    parser.add_argument("--disable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--enable-chunked-prefill", dest="enable_chunked_prefill", action="store_true", default=None)
    parser.add_argument("--disable-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    parser.add_argument("--extra-arg", action="append", help="Append a raw vLLM CLI argument. Repeat for flag/value pairs.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-instance vLLM manager for Sentrix benchmark serving.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registry profiles and local path availability.")
    p_list.set_defaults(func=command_list)

    p_cmd = sub.add_parser("cmd", help="Print the vLLM command for a profile without starting it.")
    p_cmd.add_argument("profile")
    add_runtime_options(p_cmd)
    p_cmd.set_defaults(func=command_cmd)

    p_start = sub.add_parser("start", help="Start a profile if no other managed instance is running.")
    p_start.add_argument("profile")
    add_runtime_options(p_start)
    p_start.add_argument("--replace", action="store_true", help="Stop the current managed instance first.")
    p_start.add_argument("--timeout", type=int, default=60)
    p_start.add_argument("--force", action="store_true", help="Allow stopping a pid even if /proc cmdline is unusual.")
    p_start.add_argument("--force-port", action="store_true", help="Start even if the target port is already open.")
    p_start.add_argument("--dry-run", action="store_true")
    p_start.add_argument("--wait-ready", action="store_true")
    p_start.add_argument("--ready-timeout", type=int, default=600)
    p_start.set_defaults(func=command_start)

    p_switch = sub.add_parser("switch", help="Stop the current managed instance and start the selected profile.")
    p_switch.add_argument("profile")
    add_runtime_options(p_switch)
    p_switch.add_argument("--timeout", type=int, default=60)
    p_switch.add_argument("--force", action="store_true", help="Allow stopping a pid even if /proc cmdline is unusual.")
    p_switch.add_argument("--force-port", action="store_true", help="Start even if the target port is already open.")
    p_switch.add_argument("--dry-run", action="store_true")
    p_switch.add_argument("--wait-ready", action="store_true")
    p_switch.add_argument("--ready-timeout", type=int, default=600)
    p_switch.set_defaults(func=command_switch)

    p_stop = sub.add_parser("stop", help="Stop the current managed instance, or a named legacy profile pid.")
    p_stop.add_argument("profile", nargs="?")
    p_stop.add_argument("--timeout", type=int, default=60)
    p_stop.add_argument("--kill", action="store_true")
    p_stop.add_argument("--force", action="store_true", help="Allow stopping a pid even if /proc cmdline is unusual.")
    p_stop.set_defaults(func=command_stop)

    p_status = sub.add_parser("status", help="Show the current managed instance marker.")
    p_status.add_argument("profile", nargs="?")
    p_status.add_argument("--check-http", action="store_true")
    p_status.set_defaults(func=command_status)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
