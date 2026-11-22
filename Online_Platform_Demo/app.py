# -*- coding: utf-8 -*-
"""Step6 Dual Mode V1.0.3 - unified local research and public demonstration application.

Local mode orchestrates the existing Step1-Step5 scripts; public mode reuses the bundled experiment snapshot.
Default behavior is result-first: discover and reuse valid local results; run a
script only when the requested result is missing or explicitly stale.

No API key is persisted. LLM calls are routed through the explicitly configured
local HTTP proxy (default http://127.0.0.1:4780).
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:  # Streamlit normally installs Pillow; keep a bytes fallback.
    Image = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception as exc:  # pragma: no cover - handled in UI
    go = None
    px = None
    PLOTLY_IMPORT_ERROR = str(exc)
else:
    PLOTLY_IMPORT_ERROR = ""

DUAL_MODE_VERSION = "1.0.3"
STREAMLIT_VERSION = getattr(st, "__version__", "unknown")

try:
    import networkx as nx
except Exception:
    nx = None


def button_compat(*args, **kwargs):
    """Call st.button with only arguments supported by the installed Streamlit."""
    if "type" in kwargs:
        try:
            supports_type = "type" in inspect.signature(st.button).parameters
        except (TypeError, ValueError):
            supports_type = False
        if not supports_type:
            kwargs.pop("type", None)
    return st.button(*args, **kwargs)


def divider_compat():
    """Use native divider when available; otherwise render a Markdown rule."""
    divider = getattr(st, "divider", None)
    if callable(divider):
        return divider()
    return st.markdown("---")


def _streamlit_version_tuple() -> Tuple[int, ...]:
    """Parse Streamlit's version without requiring packaging.version."""
    nums = re.findall(r"\d+", str(STREAMLIT_VERSION))
    return tuple(int(x) for x in nums[:3]) if nums else tuple()


def _supports_label_visibility() -> bool:
    """label_visibility was added after the user's Streamlit 1.12.0 baseline."""
    version = _streamlit_version_tuple()
    if version:
        return version >= (1, 16, 0)
    # Fallback for nonstandard/dev version strings.
    try:
        return "label_visibility" in inspect.signature(st.radio).parameters
    except (TypeError, ValueError):
        return False


def radio_compat(*args, **kwargs):
    """Compatibility wrapper for radio widgets on Streamlit <= 1.15."""
    if not _supports_label_visibility():
        kwargs.pop("label_visibility", None)
    return st.radio(*args, **kwargs)


def selectbox_compat(*args, **kwargs):
    """Compatibility wrapper for selectbox widgets on Streamlit <= 1.15."""
    if not _supports_label_visibility():
        kwargs.pop("label_visibility", None)
    return st.selectbox(*args, **kwargs)


def dataframe_compat(data, *args, **kwargs):
    """Render a dataframe across old/new Streamlit APIs.

    Streamlit 1.12.0 does not accept ``use_container_width`` on st.dataframe.
    Newer versions keep the requested responsive behavior. On old versions the
    argument is omitted and Streamlit uses its native automatic width.
    """
    supports_container_width = False
    try:
        params = inspect.signature(st.dataframe).parameters
        supports_container_width = "use_container_width" in params
    except (TypeError, ValueError):
        version = _streamlit_version_tuple()
        supports_container_width = bool(version and version >= (1, 13, 0))
    if not supports_container_width:
        kwargs.pop("use_container_width", None)
    return st.dataframe(data, *args, **kwargs)


# -----------------------------------------------------------------------------
# App configuration and paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Online demonstration platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = SCRIPT_DIR / "step6_workspace"
MANIFEST_DIR = WORKSPACE_DIR / "manifests"
RUNS_DIR = WORKSPACE_DIR / "decision_runs"
LOG_DIR = WORKSPACE_DIR / "logs"
STATE_PATH = WORKSPACE_DIR / "current_decision.json"
CONFIG_PATH = WORKSPACE_DIR / "config.json"

STEP1_SCRIPT = SCRIPT_DIR / "final_step1.py"
STEP2_SCRIPT = SCRIPT_DIR / "final_step2.py"
STEP3_SCRIPT = SCRIPT_DIR / "final_step3_hil.py"
STEP4_SCRIPT = SCRIPT_DIR / "final_step4_analysis.py"
STEP5_SCRIPT = SCRIPT_DIR / "final_step5_llm.py"

STEP1_OUT = SCRIPT_DIR / "step1_results_final"
STEP1_SUMMARY = STEP1_OUT / "pair_patent_summary_hybrid.csv"
STEP1_REQUIRED = [
    STEP1_SUMMARY,
    STEP1_OUT / "pair_docs_and_keywords.csv",
    STEP1_OUT / "pair_patent_candidates_hybrid_long.csv",
]
STEP3_OUT = SCRIPT_DIR / "step3_agent_output_final"
STEP4_OUT = SCRIPT_DIR / "step4_offline_paper_analysis"
STEP5_OUT = SCRIPT_DIR / "step5_chapter5_llm_ablation_tau_sweep_final"
STEP5_REPAIRED_OUT = SCRIPT_DIR / "step5_chapter5_llm_ablation_tau_sweep_final_repaired"
GAMMA_OUT = SCRIPT_DIR / "streamlit_agent_runs_full_console" / "gamma_sensitivity_runs"

# The public mode reads only the bundled, aggregated experiment snapshot.
PUBLIC_DEMO_DIR = SCRIPT_DIR / "public_demo"
PUBLIC_DATA_DIR = PUBLIC_DEMO_DIR / "data"
PUBLIC_CACHE_DIR = WORKSPACE_DIR / "public_demo_solution_cache"
PUBLIC_MODE = False

DEFAULT_PROXY = "http://127.0.0.1:4780"
DEFAULT_BASE_URL = "https://api.silra.cn"
DEFAULT_MODEL = "deepseek-v3"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    for p in (WORKSPACE_DIR, MANIFEST_DIR, RUNS_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def now_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def read_csv_any(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last = exc
    raise last or RuntimeError(f"无法读取CSV：{path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_signature(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        stt = path.stat()
        return {
            "path": str(path.resolve()),
            "exists": True,
            "size": stt.st_size,
            "mtime_ns": stt.st_mtime_ns,
            "sha256": sha256_file(path) if stt.st_size <= 30 * 1024 * 1024 else None,
        }
    items = []
    for fp in sorted(path.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in {".csv", ".xlsx", ".xls", ".xlsm"} and not fp.name.startswith("~$"):
            stt = fp.stat()
            items.append((str(fp.relative_to(path)), stt.st_size, stt.st_mtime_ns))
    raw = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return {"path": str(path.resolve()), "exists": True, "file_count": len(items), "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def fingerprint(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def pair_key(a: Any, b: Any) -> Tuple[str, str]:
    aa, bb = str(a or "").strip(), str(b or "").strip()
    return tuple(sorted((aa, bb)))


def normalize_recorded_path(raw: str) -> str:
    return str(raw or "").replace("\\", os.sep).replace("/", os.sep)


def resolve_recorded_json(raw: str) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p.resolve()
    normalized = normalize_recorded_path(raw)
    markers = ("step3_agent_output_final", "streamlit_agent_runs_full_console", "step2_results_final")
    parts = [x for x in normalized.split(os.sep) if x]
    for i, part in enumerate(parts):
        if part in markers:
            candidate = SCRIPT_DIR.joinpath(*parts[i:])
            if candidate.exists():
                return candidate.resolve()
    candidate = SCRIPT_DIR / Path(normalized).name
    return candidate.resolve() if candidate.exists() else None


def load_state() -> Dict[str, Any]:
    if PUBLIC_MODE:
        state = st.session_state.get("_step6_public_workflow_state", {}) or {}
        return dict(state) if isinstance(state, dict) else {}
    state = read_json(STATE_PATH, {}) or {}
    return state if isinstance(state, dict) else {}


def save_state(state: Dict[str, Any]) -> None:
    if PUBLIC_MODE:
        st.session_state["_step6_public_workflow_state"] = dict(state)
        return
    write_json(STATE_PATH, state)


def append_decision_log(record: Dict[str, Any]) -> None:
    if PUBLIC_MODE:
        records = list(st.session_state.get("_step6_public_decision_log", []) or [])
        records.append(dict(record))
        st.session_state["_step6_public_decision_log"] = records
        return
    path = WORKSPACE_DIR / "decision_log.csv"
    row = dict(record)
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_env(data_dir: Path, proxy: str, api_key: str, base_url: str, model: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PROJECT_DATA_DIR": str(data_dir.resolve()),
        "PROJECT_OUTPUT_DIR": str(SCRIPT_DIR.resolve()),
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "STEP6_LLM_PROXY": proxy,
        "LLM_API_KEY": api_key or "",
        "LLM_BASEURL": base_url or DEFAULT_BASE_URL,
        "LLM_MODEL": model or DEFAULT_MODEL,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def run_process(command: List[str], env: Dict[str, str], cwd: Path, label: str) -> Tuple[int, str, Path]:
    """Run a local step, stream a compact live log, and persist full output."""
    log_path = LOG_DIR / f"{now_id(label)}.log"
    lines: List[str] = []
    placeholder = st.empty()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                log.write(line)
                lines.append(line.rstrip("\r\n"))
                if len(lines) % 8 == 0:
                    placeholder.code("\n".join(lines[-16:]), language="text")
            elif proc.poll() is not None:
                break
        code = proc.wait()
    if lines:
        placeholder.code("\n".join(lines[-24:]), language="text")
    return code, "\n".join(lines), log_path


def step1_result_sha256() -> str:
    """Signature of Step1 quantitative evidence actually used by the optimizer."""
    return sha256_file(STEP1_SUMMARY) if STEP1_SUMMARY.exists() else ""


def _step3_hash_input_path(hasher, label: str, path: Optional[Path]) -> None:
    """Mirror final_step3_hil.py::_hash_input_path for legacy-cache verification."""
    hasher.update(label.encode("utf-8", errors="ignore"))
    if path is None or not path.exists():
        hasher.update(b"<MISSING>")
        return
    if path.is_file():
        hasher.update(path.name.encode("utf-8", errors="ignore"))
        hasher.update(sha256_file(path).encode("ascii"))
        return
    supported = {".csv", ".xlsx", ".xls", ".xlsm"}
    files = sorted(
        x for x in path.rglob("*")
        if x.is_file() and x.suffix.lower() in supported and not x.name.startswith("~$")
    )
    for item in files:
        rel = str(item.relative_to(path)).replace("\\", "/")
        hasher.update(rel.encode("utf-8", errors="ignore"))
        hasher.update(sha256_file(item).encode("ascii"))


def _resolve_legacy_step3_path(raw: str, bases: List[Path]) -> Optional[Path]:
    if not raw:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(raw).strip()))
    candidate = Path(expanded)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    for base in bases:
        p = (base / candidate).resolve()
        if p.exists():
            return p
    if candidate.exists():
        return candidate.resolve()
    return None


def compute_current_legacy_step3_fingerprint(review_dir: Path,
                                               data_dir: Optional[Path] = None) -> Tuple[Optional[str], str]:
    """Compute the current-input fingerprint once per app rerun."""
    summary = read_json(STEP3_OUT / "agent_run_summary.json", {}) or {}
    stored_fp = str(summary.get("step1_input_fingerprint", ""))
    if not stored_fp:
        return None, "缺少Step3历史输入指纹。"

    data_root = data_dir or DEFAULT_DATA_DIR
    work_root = SCRIPT_DIR
    raw_review = str(summary.get("batch_results_csv", "") or "")
    review_path = _resolve_legacy_step3_path(raw_review, [work_root, data_root]) or (review_dir.resolve() if review_dir.exists() else None)
    raw_summary = str(summary.get("patent_summary_csv", "") or "")
    summary_input = _resolve_legacy_step3_path(raw_summary, [work_root, data_root]) or (STEP1_SUMMARY.resolve() if STEP1_SUMMARY.exists() else None)
    raw_taxonomy = str(summary.get("improvement_taxonomy_xlsx", "") or "")
    taxonomy_path = _resolve_legacy_step3_path(raw_taxonomy, [work_root, data_root])
    raw_step2 = str(summary.get("step2_script", "") or "")
    step2_path = _resolve_legacy_step3_path(raw_step2, [work_root]) or (STEP2_SCRIPT.resolve() if STEP2_SCRIPT.exists() else None)

    if review_path is None or summary_input is None or step2_path is None:
        return None, "当前评论数据、Step1汇总或Step2脚本缺失，无法完成历史指纹对照。"

    import hashlib as _hashlib
    h = _hashlib.sha256()
    _step3_hash_input_path(h, "review_data", review_path)
    _step3_hash_input_path(h, "step1_patent_summary", summary_input)
    if taxonomy_path is not None and taxonomy_path.exists():
        _step3_hash_input_path(h, "improvement_taxonomy", taxonomy_path)
    if step2_path.exists():
        _step3_hash_input_path(h, "step2_script", step2_path)
    run_settings = {
        "gamma": float(summary.get("configured_gamma", summary.get("gamma", 1.0))),
        "time_limit_sec": int(summary.get("time_limit_sec", 0)),
        "exact_cand_pool": int(summary.get("exact_cand_pool", 0)),
    }
    h.update(json.dumps(run_settings, sort_keys=True).encode("utf-8"))
    return h.hexdigest(), ""


def verify_legacy_step3_binding(json_path: Path, review_dir: Path,
                                data_dir: Optional[Path] = None,
                                current_fingerprint: Optional[str] = None,
                                fingerprint_error: str = "") -> Tuple[str, str]:
    """Verify a pre-Step6 Step2 result using Step3's own input/cache fingerprints.

    Older Step2 outputs did not have step6_binding.json, but the project archive
    contains agent_run_summary.json and _step3_cache_manifest.json. These carry
    Step3's SHA-256 input fingerprint and the exact Step2-result hash.
    """
    summary = read_json(STEP3_OUT / "agent_run_summary.json", {}) or {}
    manifest = read_json(json_path.parent / "_step3_cache_manifest.json", {}) or {}
    if not summary.get("step1_input_fingerprint") or not manifest.get("input_fingerprint"):
        return "legacy_unverified", "缺少Step3历史输入指纹或缓存清单，仍需人工核对。"
    stored_fp = str(summary.get("step1_input_fingerprint", ""))
    if str(manifest.get("input_fingerprint", "")) != stored_fp:
        return "stale", "Step3汇总与该优化结果的历史输入指纹不一致。"
    if str(manifest.get("result_json_sha256", "")) != sha256_file(json_path):
        return "stale", "Step2结果文件与Step3缓存清单中的SHA-256不一致。"

    try:
        result = load_solution_json(json_path)
        meta = result.get("meta", {}) or {}
        if (int(manifest.get("k_select")) != int(meta.get("k_select"))
                or int(manifest.get("max_edges_per_node")) != int(meta.get("max_edges_per_node"))
                or abs(float(manifest.get("gamma")) - float(meta.get("gamma"))) > 1e-9):
            return "stale", "Step3缓存清单中的K/B/γ与Step2结果不一致。"
        if (int(manifest.get("time_limit_sec", -1)) != int(summary.get("time_limit_sec", -2))
                or int(manifest.get("exact_cand_pool", -1)) != int(summary.get("exact_cand_pool", -2))):
            return "stale", "Step3缓存清单中的求解设置与历史运行记录不一致。"
    except Exception as exc:
        return "stale", f"无法核验Step2结果元数据：{exc}"

    if current_fingerprint is None:
        current_fingerprint, fingerprint_error = compute_current_legacy_step3_fingerprint(review_dir, data_dir)
    if current_fingerprint is None:
        return "legacy_unverified", fingerprint_error or "无法计算当前输入指纹，仍需人工核对。"
    if current_fingerprint == stored_fp:
        return "verified_legacy", "Step3历史输入指纹、当前输入文件及Step2结果哈希一致；仅缺少Step6界面绑定记录。"
    return "stale", "当前评论数据、Step1专利汇总、技术分类或Step2脚本与历史Step3输入指纹不一致。"


def step1_fingerprint(data_dir: Path, review_dir: Path, patent_file: Path) -> Dict[str, Any]:
    return {
        "step1_script": file_signature(STEP1_SCRIPT),
        "review_input": file_signature(review_dir),
        "patent_file": file_signature(patent_file),
        "data_dir": str(data_dir.resolve()),
    }


def step1_status(data_dir: Path, review_dir: Path, patent_file: Path) -> Dict[str, Any]:
    missing = [str(p) for p in STEP1_REQUIRED if not p.exists()]
    manifest_path = MANIFEST_DIR / "step1_manifest.json"
    current_fp = fingerprint(step1_fingerprint(data_dir, review_dir, patent_file))
    manifest = read_json(manifest_path, {}) or {}
    if missing:
        status, label = "missing", "需要运行 Step1"
    elif manifest.get("fingerprint") == current_fp and manifest.get("verification") == "produced_by_step6":
        status, label = "verified", "已有结果 · 输入版本一致"
    elif manifest.get("fingerprint") == current_fp:
        status, label = "legacy", "历史结果 · 已人工确认来源"
    elif manifest:
        status, label = "stale", "已有结果 · 输入已变化"
    else:
        status, label = "legacy", "已有历史结果 · 版本待核验"
    return {"status": status, "label": label, "missing": missing, "fingerprint": current_fp, "manifest": manifest}


def step1_cache_ready() -> bool:
    expected = 0
    pair_list = STEP1_OUT / "pair_issue_distribution.csv"
    if pair_list.exists():
        try:
            expected = len(read_csv_any(pair_list))
        except Exception:
            expected = 0
    if expected <= 0:
        expected = len(build_candidate_edges())
    counts = []
    for name in ("llm_rerank_cache.json", "llm_need_tech_align_cache.json"):
        data = read_json(STEP1_OUT / name, {})
        if not isinstance(data, dict) or not data:
            return False
        counts.append(len(data))
    return all(n >= expected for n in counts) if expected > 0 else all(n > 0 for n in counts)


def prepare_proxy_step1() -> Path:
    """Create an isolated Step1 copy that explicitly applies the configured proxy.

    The original Step1 client disables requests' environment proxy discovery, so
    merely setting HTTPS_PROXY is insufficient. This runtime copy adds proxies
    directly to its requests.Session without changing the user's original file.
    """
    if not STEP1_SCRIPT.exists():
        raise FileNotFoundError(f"未找到 Step1 脚本：{STEP1_SCRIPT}")
    runtime_dir = SCRIPT_DIR / ".step6_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / "final_step1_step6_proxy.py"
    source = STEP1_SCRIPT.read_text(encoding="utf-8")
    if "STEP1_REVIEW_INPUT_DIR" not in source:
        source = source.replace(
            'REVIEW_INPUT_DIR = _data_path("统一创新单元final")',
            'REVIEW_INPUT_DIR = os.getenv("STEP1_REVIEW_INPUT_DIR", _data_path("统一创新单元final"))',
            1,
        )
        source = source.replace(
            'PATENT_XLSX = _data_path("专利.xlsx")',
            'PATENT_XLSX = os.getenv("STEP1_PATENT_XLSX", _data_path("专利.xlsx"))',
            1,
        )
    # Compact Step1 already passes proxies explicitly. Avoid patching it again.
    has_explicit_proxy = "proxies=proxies" in source or "proxies = {" in source
    if not has_explicit_proxy and "STEP6_LLM_PROXY" not in source:
        old = """                with requests.Session() as _s:\n                    _s.trust_env = False\n                    r = _s.post(\n"""
        new = """                with requests.Session() as _s:\n                    _s.trust_env = False\n                    _proxy = os.getenv(\"STEP6_LLM_PROXY\", os.getenv(\"HTTPS_PROXY\", \"\")).strip()\n                    if _proxy:\n                        _s.proxies.update({\"http\": _proxy, \"https\": _proxy})\n                    r = _s.post(\n"""
        if old not in source:
            raise RuntimeError("无法安全生成代理适配版 Step1：未找到预期的 requests.Session 调用代码。原脚本未被修改。")
        source = source.replace(old, new, 1)
    target.write_text(source, encoding="utf-8")
    return target


def prepare_fixed_step3() -> Path:
    """Create a runtime Step3 copy with a one-(K,B) execution mode.

    Existing Step3 remains untouched. The added mode is used only when the user
    requests one exact parameter combination; it still calls the original Step2.
    """
    if not STEP3_SCRIPT.exists():
        raise FileNotFoundError(f"未找到 Step3 脚本：{STEP3_SCRIPT}")
    runtime_dir = SCRIPT_DIR / ".step6_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / "final_step3_hil_v22.py"
    source = STEP3_SCRIPT.read_text(encoding="utf-8")
    if "AGENT_FIXED_KB_ONLY" not in source:
        old = '''    allowed_b_values = sorted(_env_int_list("AGENT_ALLOWED_B_VALUES", [2, 3, 4]))\n    node_count, edge_count = prescan_review_graph(batch_results_csv, work_dir)\n\n    k_step = _env_int("AGENT_K_STEP", 2)\n    k_count = _env_int("AGENT_K_COUNT_PER_B", 5)\n    allowed_k_values_by_b = build_allowed_k_values_by_b(\n        node_count,\n        edge_count,\n        allowed_b_values,\n        step=k_step,\n        count=k_count,\n    )\n\n    initial_b = _env_int("AGENT_INITIAL_B", 3)\n    if initial_b not in allowed_b_values:\n        initial_b = allowed_b_values[0]\n    requested_initial_k = _env_int("AGENT_INITIAL_K", 10)\n    initial_k = choose_initial_k(requested_initial_k, initial_b, allowed_k_values_by_b)\n'''
        new = '''    allowed_b_values = sorted(_env_int_list("AGENT_ALLOWED_B_VALUES", [2, 3, 4]))\n    node_count, edge_count = prescan_review_graph(batch_results_csv, work_dir)\n\n    k_step = _env_int("AGENT_K_STEP", 2)\n    k_count = _env_int("AGENT_K_COUNT_PER_B", 5)\n    requested_initial_k = _env_int("AGENT_INITIAL_K", 10)\n    fixed_kb_only = _env_bool("AGENT_FIXED_KB_ONLY", False)\n    if fixed_kb_only:\n        initial_b = _env_int("AGENT_INITIAL_B", 3)\n        max_feasible_k = min(edge_count, (node_count * initial_b) // 2)\n        if initial_b <= 0 or requested_initial_k <= 0 or requested_initial_k > max_feasible_k:\n            raise ValueError(f\"固定 K/B 不可行：K={requested_initial_k}, B={initial_b}, 理论 K 上限={max_feasible_k}\")\n        allowed_b_values = [initial_b]\n        allowed_k_values_by_b = {initial_b: [requested_initial_k]}\n        initial_k = requested_initial_k\n    else:\n        allowed_k_values_by_b = build_allowed_k_values_by_b(\n            node_count, edge_count, allowed_b_values, step=k_step, count=k_count\n        )\n        initial_b = _env_int("AGENT_INITIAL_B", 3)\n        if initial_b not in allowed_b_values:\n            initial_b = allowed_b_values[0]\n        initial_k = choose_initial_k(requested_initial_k, initial_b, allowed_k_values_by_b)\n'''
        if old not in source:
            raise RuntimeError("无法安全生成固定 K/B 模式 Step3：原始参数构建代码与预期版本不一致。原脚本未被修改。")
        source = source.replace(old, new, 1)
        old_iter = '''    pairs: List[Tuple[int, int]] = []\n    preferred = (int(state.initial_k), int(state.initial_b))\n'''
        new_iter = '''    pairs: List[Tuple[int, int]] = []\n    preferred = (int(state.initial_k), int(state.initial_b))\n    if _env_bool("AGENT_FIXED_KB_ONLY", False):\n        return [preferred]\n'''
        if old_iter not in source:
            raise RuntimeError("无法安全生成固定 K/B 模式 Step3：未找到参数枚举入口。原脚本未被修改。")
        source = source.replace(old_iter, new_iter, 1)
    target.write_text(source, encoding="utf-8")
    return target


def build_candidate_edges() -> pd.DataFrame:
    if PUBLIC_MODE:
        public_edges = PUBLIC_DATA_DIR / "candidate_edges.csv"
        if not public_edges.exists():
            return pd.DataFrame()
        df = read_csv_any(public_edges)
        if df.empty:
            return df
        ca = next((c for c in ("product_a", "u", "product1", "p1") if c in df.columns), None)
        cb = next((c for c in ("product_b", "v", "product2", "p2") if c in df.columns), None)
        if not ca or not cb:
            return pd.DataFrame()
        out = df.copy()
        out["product_a"] = out[ca].astype(str).str.strip()
        out["product_b"] = out[cb].astype(str).str.strip()
        out = out[(out.product_a != "") & (out.product_b != "") & (out.product_a != out.product_b)].copy()
        if "edge_key" not in out.columns:
            out["edge_key"] = ["||".join(pair_key(a, b)) for a, b in zip(out.product_a, out.product_b)]
        if "pair_freq" not in out.columns and "freq" in out.columns:
            out["pair_freq"] = pd.to_numeric(out["freq"], errors="coerce").fillna(1.0)
        return out.drop_duplicates("edge_key").reset_index(drop=True)

    if not STEP1_SUMMARY.exists():
        return pd.DataFrame()
    df = read_csv_any(STEP1_SUMMARY)
    ca = next((c for c in ("product_a", "u", "product1", "p1") if c in df.columns), None)
    cb = next((c for c in ("product_b", "v", "product2", "p2") if c in df.columns), None)
    if not ca or not cb:
        return pd.DataFrame()
    out = df.copy()
    out["product_a"] = out[ca].astype(str).str.strip()
    out["product_b"] = out[cb].astype(str).str.strip()
    out = out[(out.product_a != "") & (out.product_b != "") & (out.product_a != out.product_b)].copy()
    out["edge_key"] = ["|".join(pair_key(a, b)) for a, b in zip(out.product_a, out.product_b)]
    return out.drop_duplicates("edge_key").reset_index(drop=True)


def load_solution_json(path: Path) -> Dict[str, Any]:
    data = read_json(path, {})
    if not isinstance(data, dict) or "meta" not in data or "selected_pairs" not in data:
        raise ValueError(f"不是有效的Step2精确优化结果：{path}")
    return data


def scan_cached_solutions(current_step1_fingerprint: str = "", review_dir: Optional[Path] = None, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    roots = [STEP3_OUT, GAMMA_OUT]
    legacy_current_fp, legacy_fp_error = (None, "未提供评论数据目录。")
    if review_dir is not None:
        legacy_current_fp, legacy_fp_error = compute_current_legacy_step3_fingerprint(review_dir, data_dir)
    records: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("edgeset_selection_exact.json"):
            if any(part.startswith("_step2_all_candidate_edges") for part in path.parts):
                continue
            try:
                data = load_solution_json(path)
                meta = data.get("meta", {})
                k = safe_int(meta.get("k_select"), -1)
                b = safe_int(meta.get("max_edges_per_node"), -1)
                gamma = safe_float(meta.get("gamma"), float("nan"))
                solver = meta.get("solver", {}) or {}
                if k <= 0 or b <= 0 or pd.isna(gamma):
                    continue
                binding = read_json(path.parent / "step6_binding.json", {}) or {}
                if binding:
                    binding_ok = (
                        binding.get("step1_fingerprint") == current_step1_fingerprint
                        and binding.get("step1_result_sha256") == step1_result_sha256()
                        and binding.get("step2_script_sha256") == (sha256_file(STEP2_SCRIPT) if STEP2_SCRIPT.exists() else "")
                        and binding.get("step3_script_sha256") == (sha256_file(STEP3_SCRIPT) if STEP3_SCRIPT.exists() else "")
                    )
                    input_status = "verified" if binding_ok else "stale"
                elif review_dir is not None:
                    input_status, _ = verify_legacy_step3_binding(path, review_dir, data_dir, legacy_current_fp, legacy_fp_error)
                else:
                    input_status = "legacy_unverified"
                rec = {
                    "path": str(path.resolve()),
                    "csv_path": str(path.with_suffix(".csv").resolve()),
                    "k": k,
                    "b": b,
                    "gamma": float(gamma),
                    "weight": safe_float(solver.get("best_weight")),
                    "optimal": bool(solver.get("optimal", False)),
                    "selected_count": len(data.get("selected_pairs", []) or []),
                    "mtime": path.stat().st_mtime,
                    "source": "Gamma情景" if "gamma_sensitivity" in str(path) else "Step3历史结果",
                    "input_status": input_status,
                }
                records[str(path.resolve())] = rec
            except Exception:
                continue
    return sorted(records.values(), key=lambda x: (x["k"], x["b"], x["gamma"], x["mtime"]))


def find_cached_solution(k: int, b: int, gamma: float, cache: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    matches = [x for x in cache if x["k"] == int(k) and x["b"] == int(b) and abs(x["gamma"] - float(gamma)) < 1e-8 and x.get("input_status") != "stale"]
    if not matches:
        return None
    # Prefer canonical Step3 results over older temporary copies when both match.
    status_priority = {"verified": 0, "verified_legacy": 1, "legacy_unverified": 2, "stale": 3}
    matches.sort(key=lambda x: (status_priority.get(x.get("input_status"), 4),
                               0 if x["source"] == "Step3历史结果" else 1, -x["mtime"]))
    return matches[0]


def default_solution(cache: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    summary_path = STEP3_OUT / "agent_run_summary.json"
    summary = read_json(summary_path, {}) or {}
    resolved = resolve_recorded_json(str(summary.get("final_json_path", "")))
    if resolved:
        try:
            cached_record = next((x for x in cache if Path(x["path"]).resolve() == resolved.resolve()), None)
            if cached_record and cached_record.get("input_status") == "stale":
                raise ValueError("历史Step2结果对应的Step1输入已变化")
            data = load_solution_json(resolved)
            meta = data["meta"]
            return {
                "path": str(resolved),
                "csv_path": str(resolved.with_suffix(".csv")),
                "k": safe_int(meta.get("k_select")),
                "b": safe_int(meta.get("max_edges_per_node")),
                "gamma": safe_float(meta.get("gamma"), 1.0),
                "weight": safe_float((meta.get("solver") or {}).get("best_weight")),
                "optimal": bool((meta.get("solver") or {}).get("optimal", False)),
                "selected_count": len(data.get("selected_pairs", []) or []),
                "mtime": resolved.stat().st_mtime,
                "source": "当前历史推荐",
            }
        except Exception:
            pass
    usable = [x for x in cache if x.get("input_status") != "stale"]
    if usable:
        baseline = find_cached_solution(12, 4, 1.0, usable)
        return baseline or max(usable, key=lambda x: (x["weight"], x["mtime"]))
    return None


def prepare_public_demo_data() -> None:
    """Materialize Step6-native JSON wrappers for the bundled experiment snapshot."""
    cache_file = PUBLIC_DATA_DIR / "cached_solutions.json"
    candidate_file = PUBLIC_DATA_DIR / "candidate_edges.csv"
    if not cache_file.exists() or not candidate_file.exists():
        raise FileNotFoundError("公开演示数据不完整：缺少候选关系或参数方案数据。")
    cached = read_json(cache_file, {}) or {}
    if not isinstance(cached, dict) or not cached:
        raise ValueError("公开演示参数方案文件为空或格式不正确。")
    PUBLIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = read_csv_any(candidate_file)
    default_json = None
    for key, record in cached.items():
        if not isinstance(record, dict):
            continue
        k = safe_int(record.get("k"), -1)
        b = safe_int(record.get("b"), -1)
        gamma = safe_float(record.get("gamma"), float("nan"))
        selected = record.get("selected_pairs", []) or []
        if k <= 0 or b <= 0 or pd.isna(gamma) or not selected:
            continue
        degree_usage: Dict[str, int] = {}
        for pair in selected:
            a = str(pair.get("product_a", "")).strip()
            bb = str(pair.get("product_b", "")).strip()
            if a:
                degree_usage[a] = degree_usage.get(a, 0) + 1
            if bb:
                degree_usage[bb] = degree_usage.get(bb, 0) + 1
        solution = {
            "meta": {
                "k_select": k,
                "max_edges_per_node": b,
                "gamma": gamma,
                "candidate_pool": len(candidate),
                "degree_usage": degree_usage,
                "solver": {
                    "best_weight": safe_float(record.get("objective")),
                    "optimal": bool(record.get("optimal", False)),
                    "timed_out": False,
                    "status": "optimal" if bool(record.get("optimal", False)) else "available_snapshot",
                },
                "public_snapshot": True,
            },
            "selected_pairs": selected,
        }
        folder = PUBLIC_CACHE_DIR / f"public_k{k}_b{b}_g{gamma:.2f}"
        folder.mkdir(parents=True, exist_ok=True)
        json_path = folder / "edgeset_selection_exact.json"
        write_json(json_path, solution)
        try:
            pd.DataFrame(selected).to_csv(json_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        except Exception:
            pass
        if k == 12 and b == 4 and abs(gamma - 1.0) < 1e-8:
            default_json = json_path
    if default_json is None:
        raise ValueError("公开演示数据中没有 K=12、B=4、γ=1.0 的基准方案。")
    write_json(PUBLIC_CACHE_DIR / "agent_run_summary.json", {"final_json_path": str(default_json.resolve())})


def public_step1_info() -> Dict[str, Any]:
    candidate_file = PUBLIC_DATA_DIR / "candidate_edges.csv"
    if not candidate_file.exists():
        return {"status": "missing", "label": "需求与技术证据不可用", "missing": [str(candidate_file)], "fingerprint": ""}
    return {
        "status": "verified",
        "label": "需求与技术证据已加载",
        "missing": [],
        "fingerprint": sha256_file(candidate_file),
        "manifest": {"source": "bundled_public_experiment_snapshot"},
    }


def public_solution_summary(solution: Dict[str, Any]) -> Dict[str, Any]:
    meta = solution.get("meta", {}) or {}
    solver = meta.get("solver", {}) or {}
    selected = solution.get("selected_pairs", []) or []
    nodes = set()
    for row in selected:
        nodes.update([str(row.get("product_a", "")), str(row.get("product_b", ""))])
    nodes.discard("")
    return {
        "k": safe_int(meta.get("k_select")),
        "b": safe_int(meta.get("max_edges_per_node")),
        "gamma": safe_float(meta.get("gamma"), 1.0),
        "objective": safe_float(solver.get("best_weight")),
        "selected_count": len(selected),
        "unique_nodes": len(nodes),
        "max_node_usage": max((safe_int(v) for v in (meta.get("degree_usage") or {}).values()), default=0),
    }


def public_gamma_figure(k: int, b: int):
    """Return the bundled gamma sensitivity curve for its observed K/B scope."""
    if go is None or (int(k), int(b)) != (12, 4):
        return None
    path = PUBLIC_DATA_DIR / "gamma_sensitivity.csv"
    if not path.exists():
        return None
    df = read_csv_any(path).sort_values("gamma")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["gamma"], y=df["total_weight"], mode="lines+markers",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=8, color="#2563EB"),
        name="累计目标值",
    ))
    fig.update_layout(
        template="plotly_white", height=300,
        margin=dict(l=28, r=18, t=10, b=38),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis_title="技术增益系数 γ", yaxis_title="累计目标值",
        font=dict(family="Arial, Microsoft YaHei, sans-serif", color="#344054", size=11),
        hovermode="x unified",
    )
    return fig


def render_public_case_analysis(active_json: Dict[str, Any]) -> None:
    selected = pd.DataFrame(active_json.get("selected_pairs", []) or [])
    candidate = build_candidate_edges()
    summary = public_solution_summary(active_json)
    metric_row([
        ("入选产品关系", summary["selected_count"], f"K = {summary['k']}"),
        ("协同优先级总值", f"{summary['objective']:.4f}", f"γ = {summary['gamma']:.2f}"),
        ("覆盖产品节点", summary["unique_nodes"], "当前入选组合"),
        ("单产品最大参与", summary["max_node_usage"], f"B = {summary['b']}"),
    ])
    st.markdown("**最终入选协同改进网络**")
    left, right = st.columns([1.08, 0.92])
    with left:
        fig = network_figure(candidate, selected, "优化网络", height=430)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
    with right:
        node_rows = []
        for node, degree in (active_json.get("meta", {}).get("degree_usage", {}) or {}).items():
            incident = selected[
                (selected["product_a"].astype(str) == str(node)) |
                (selected["product_b"].astype(str) == str(node))
            ] if not selected.empty else pd.DataFrame()
            node_rows.append({
                "产品节点": paper_product_label(node),
                "参与关系数": safe_int(degree),
                "关联优先级": round(pd.to_numeric(incident.get("combined_weight", pd.Series(dtype=float)), errors="coerce").sum(), 4),
            })
        node_df = pd.DataFrame(node_rows).sort_values("参与关系数", ascending=False) if node_rows else pd.DataFrame()
        st.markdown("**产品节点参与情况**")
        if not node_df.empty:
            fig = px.bar(node_df.sort_values("参与关系数"), x="参与关系数", y="产品节点", orientation="h",
                         labels={"参与关系数": "参与关系数", "产品节点": "产品节点"})
            fig.update_layout(height=430, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("**最终入选产品关系及改进方向**")
    explanation = decision_explanation_table(selected)
    dataframe_compat(explanation, use_container_width=True, height=390)
    st.download_button(
        "下载当前方案的协同改进关系",
        data=explanation.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"selected_pairs_K{summary['k']}_B{summary['b']}.csv",
        mime="text/csv",
    )

    issue_path = PUBLIC_DATA_DIR / "issue_distribution.csv"
    if issue_path.exists() and not selected.empty:
        issues = read_csv_any(issue_path)
        selected_keys = {
            "||".join(pair_key(row.get("product_a", ""), row.get("product_b", "")))
            for _, row in selected.iterrows()
        }
        issues = issues[issues["pair_key"].astype(str).isin(selected_keys)].copy()
        issues = issues[issues["issue"].astype(str).str.strip() != "满意"].copy()
        if not issues.empty:
            direction = (issues.groupby(["scenario", "issue"], dropna=False)
                         .agg(方向严重度=("issue_severity_sum", "sum"),
                              关联反馈数=("issue_count", "sum"))
                         .reset_index()
                         .sort_values("方向严重度", ascending=False).head(15))
            direction["改进方向"] = direction["scenario"].astype(str) + "｜" + direction["issue"].astype(str)
            st.markdown("**入选关系对应的具体改进方向**")
            fig = px.bar(direction.sort_values("方向严重度"), x="方向严重度", y="改进方向",
                         orientation="h", labels={"方向严重度": "方向严重度", "改进方向": "改进方向"})
            fig.update_layout(height=max(360, 26 * len(direction)), margin=dict(l=5, r=5, t=10, b=5),
                              paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            dataframe_compat(direction[["scenario", "issue", "方向严重度", "关联反馈数"]],
                             use_container_width=True, height=260)

    ipc_counter: Dict[str, int] = {}
    for value in selected.get("ipc_primary", pd.Series(dtype=str)).dropna().astype(str):
        try:
            parsed = json.loads(value) if value.startswith("[") else [value]
        except Exception:
            parsed = [value]
        for code in parsed:
            code = str(code).strip()
            if code:
                ipc_counter[code] = ipc_counter.get(code, 0) + 1
    if ipc_counter:
        ipc_df = pd.DataFrame([{"IPC小类": key, "关联产品对数": val} for key, val in ipc_counter.items()])
        ipc_df = ipc_df.sort_values("关联产品对数", ascending=False)
        st.markdown("**入选关系的IPC技术分类**")
        fig = px.bar(ipc_df.head(15).sort_values("关联产品对数"), x="关联产品对数", y="IPC小类",
                     orientation="h", labels={"关联产品对数": "关联产品关系数", "IPC小类": "IPC小类"})
        fig.update_layout(height=max(330, 24 * min(15, len(ipc_df))), margin=dict(l=5, r=5, t=10, b=5),
                          paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_public_robustness() -> None:
    gamma_path = PUBLIC_DATA_DIR / "gamma_sensitivity.csv"
    baseline_path = PUBLIC_DATA_DIR / "baseline_comparison.csv"
    llm_path = PUBLIC_DATA_DIR / "llm_robustness_summary.csv"
    trials_path = PUBLIC_DATA_DIR / "parameter_trials.csv"

    if gamma_path.exists():
        gamma_df = read_csv_any(gamma_path).sort_values("gamma")
        st.markdown("**技术增益系数敏感性**")
        left, right = st.columns(2)
        with left:
            fig = px.line(gamma_df, x="gamma", y="total_weight", markers=True,
                          labels={"gamma": "技术增益系数 γ", "total_weight": "累计目标值"})
            fig.update_traces(line_color="#2563EB", line_width=3, marker_size=8)
            fig.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            fig = px.line(gamma_df, x="gamma", y="overlap_rate_with_baseline", markers=True,
                          labels={"gamma": "技术增益系数 γ", "overlap_rate_with_baseline": "与基准方案的边集重合率"})
            fig.update_traces(line_color="#D97706", line_width=3, marker_size=8)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
            fig.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        show = gamma_df.rename(columns={
            "gamma": "γ", "total_weight": "累计目标值", "avg_edge_weight": "单位边平均权重",
            "unique_nodes": "覆盖产品节点", "max_node_usage": "最大节点参与数",
            "overlap_with_baseline": "重合关系数", "overlap_rate_with_baseline": "边集重合率",
        })
        cols = [c for c in ["γ", "累计目标值", "单位边平均权重", "覆盖产品节点", "最大节点参与数", "重合关系数", "边集重合率"] if c in show.columns]
        dataframe_compat(show[cols], use_container_width=True, height=270)

    if baseline_path.exists():
        baseline = read_csv_any(baseline_path)
        st.markdown("**基准方法比较**")
        fig = px.bar(baseline, x="method", y="total_weight_under_proposed_metric", color="method",
                     labels={"method": "选择方法", "total_weight_under_proposed_metric": "本文综合权重下的累计值"})
        fig.update_layout(height=350, showlegend=False, margin=dict(l=5, r=5, t=10, b=5),
                          paper_bgcolor="#FFFFFF", xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        show = baseline.rename(columns={
            "method": "对照方法", "selected_count": "入选数量",
            "total_weight_under_proposed_metric": "累计权重", "avg_weight_per_edge": "单位边平均权重",
            "unique_nodes": "覆盖节点数", "max_node_usage": "最大节点参与数",
            "max_node_share": "最大节点占比", "overlap_with_proposed_edges": "重合关系数",
            "overlap_rate": "边集重合率",
        })
        dataframe_compat(show, use_container_width=True, height=260)

    if trials_path.exists():
        trials = read_csv_any(trials_path).sort_values(["B", "K"])
        st.markdown("**K/B参数组合比较**")
        fig = px.scatter(trials, x="K", y="objective", color="B", size="selected_count",
                         hover_data=["avg_weight_per_edge", "unique_nodes", "max_node_usage", "optimal"],
                         labels={"K": "产品对数量 K", "objective": "累计目标值", "B": "单产品参与上限"})
        fig.update_layout(height=350, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if llm_path.exists():
        llm = read_csv_any(llm_path)
        main = llm[llm["analysis_type"].astype(str) == "main_with_llm"].head(1)
        ablation = llm[llm["analysis_type"].astype(str) == "no_llm_ablation"].head(1)
        perturb = llm[llm["analysis_type"].astype(str) == "llm_perturb"].copy()
        if not main.empty and not ablation.empty:
            st.markdown("**大语言模型语义校验的消融比较**")
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_row([("主模型累计目标值", f"{safe_float(main.iloc[0]['mean_best_weight']):.4f}", "保留语义校验")])
            with c2:
                metric_row([("无LLM消融累计值", f"{safe_float(ablation.iloc[0]['mean_best_weight']):.4f}", "移除语义校验")])
            with c3:
                metric_row([("无LLM边集重合率", f"{safe_float(ablation.iloc[0]['mean_overlap_rate']):.1%}", "相对主模型")])
        if not perturb.empty:
            st.markdown("**语义扰动下的方案稳定性**")
            perturb = perturb.sort_values("tau")
            fig = px.bar(perturb, x=perturb["tau"].astype(str), y="mean_best_weight",
                         labels={"x": "扰动温度 τ", "mean_best_weight": "平均累计目标值"})
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            show_cols = [c for c in ["tau", "mean_best_weight", "min_best_weight", "max_best_weight",
                                     "mean_overlap_rate", "mean_tech_strength_cv"] if c in perturb.columns]
            show = perturb[show_cols].rename(columns={
                "tau": "扰动温度 τ", "mean_best_weight": "平均目标值",
                "min_best_weight": "最小目标值", "max_best_weight": "最大目标值",
                "mean_overlap_rate": "平均边集重合率",
                "mean_tech_strength_cv": "技术强度CV均值",
            })
            dataframe_compat(show, use_container_width=True, height=260)


def make_summary_for_solution(json_path: Path, run_dir: Path, review_dir: Path, patent_summary: Path) -> Path:
    data = load_solution_json(json_path)
    meta = data.get("meta", {})
    solver = meta.get("solver", {}) or {}
    selected = data.get("selected_pairs", []) or []
    degree = meta.get("degree_usage", {}) or {}
    used = [safe_int(v) for v in degree.values() if safe_int(v) > 0]
    max_usage = max(used) if used else 0
    unique_nodes = len(used)
    max_share = max_usage / max(1, len(selected))
    k, b = safe_int(meta.get("k_select")), safe_int(meta.get("max_edges_per_node"))
    gamma = safe_float(meta.get("gamma"), 1.0)
    edge_df = build_candidate_edges()
    node_count = len(set(edge_df.get("product_a", pd.Series(dtype=str)).astype(str)) | set(edge_df.get("product_b", pd.Series(dtype=str)).astype(str)))
    trial = {
        "trial_id": 1,
        "k": k,
        "b": b,
        "time_limit_sec": 0,
        "step2_out_dir": str(json_path.parent),
        "status": "success",
        "reason": "reused_existing_result" if "step6_runs" not in str(json_path) else "computed_by_step6_workflow",
        "best_weight": safe_float(solver.get("best_weight")),
        "optimal": bool(solver.get("optimal", False)),
        "timed_out": bool(solver.get("timed_out", False)),
        "selected_count": len(selected),
        "unique_nodes": unique_nodes,
        "max_node_usage": max_usage,
        "max_node_share": max_share,
        "json_path": str(json_path.resolve()),
        "started_at": None,
        "finished_at": None,
        "duration_sec": safe_float(solver.get("time_sec")),
    }
    summary = {
        "goal": "Human-guided ecosystem product improvement selection",
        "run_step1": False,
        "step1_script": str(STEP1_SCRIPT),
        "step2_script": str(STEP2_SCRIPT),
        "work_dir": str(SCRIPT_DIR),
        "agent_out_dir": str(run_dir),
        "requested_initial_k": k,
        "initial_k": k,
        "initial_b": b,
        "time_limit_sec": 0,
        "selection_policy": "human_specified_k_b",
        "allowed_b_values": [b],
        "allowed_k_values_by_b": {str(b): [k]},
        "node_count": node_count,
        "edge_count": len(edge_df),
        "batch_results_csv": str(review_dir.resolve()),
        "patent_summary_csv": str(patent_summary.resolve()),
        "gamma": gamma,
        "exact_cand_pool": safe_int(meta.get("candidate_pool"), 0),
        "plot_top_edges": 220,
        "improvement_taxonomy_xlsx": str(DEFAULT_DATA_DIR / "互联产品改进类别.xlsx"),
        "disable_step2_llm_plan": True,
        "accepted_trial_id": 1,
        "final_json_path": str(json_path.resolve()),
        "planner_trace_path": str(run_dir / "planner_trace.jsonl"),
        "trials": [trial],
    }
    summary_path = run_dir / "agent_run_summary.json"
    write_json(summary_path, summary)
    return summary_path


def activate_solution(json_path: Path, review_dir: Path, patent_summary: Path, source: str, summary_path: Optional[Path] = None, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    data = load_solution_json(json_path)
    meta = data.get("meta", {})
    run_dir = RUNS_DIR / ("reuse_" + sha256_file(json_path)[:12])
    run_dir.mkdir(parents=True, exist_ok=True)
    if summary_path is None or not summary_path.exists():
        summary_path = make_summary_for_solution(json_path, run_dir, review_dir, patent_summary)
    state = load_state()
    result_binding = read_json(json_path.parent / "step6_binding.json", {}) or {}
    current_step1_manifest = read_json(MANIFEST_DIR / "step1_manifest.json", {}) or {}
    if PUBLIC_MODE:
        input_binding_status = "demo_snapshot"
        input_binding_reason = "该方案与公开演示数据包中的实验数据配套。"
    elif result_binding:
        current_fp = current_step1_manifest.get("fingerprint", "")
        if not current_fp and "data_dir" in globals() and "patent_file" in globals():
            current_fp = fingerprint(step1_fingerprint(data_dir or DEFAULT_DATA_DIR, review_dir, patent_file))
        input_binding_status = "verified" if (
            result_binding.get("step1_fingerprint") == current_fp
            and result_binding.get("step1_result_sha256") == step1_result_sha256()
        ) else "stale"
        input_binding_reason = "已有Step6绑定记录与当前Step1结果一致。" if input_binding_status == "verified" else "已有Step6绑定记录与当前Step1结果不一致。"
    else:
        input_binding_status, input_binding_reason = verify_legacy_step3_binding(json_path, review_dir, data_dir)
    state.update({
        "active_solution_path": str(json_path.resolve()),
        "active_solution_csv": str(json_path.with_suffix(".csv").resolve()),
        "active_summary_path": str(summary_path.resolve()),
        "active_solution_hash": sha256_file(json_path),
        "active_solution_source": source,
        "active_input_binding_status": input_binding_status,
        "active_input_binding_reason": input_binding_reason,
        "step4_out_dir": "",
        "step5_out_dir": "",
        "active_params": {
            "k": safe_int(meta.get("k_select")),
            "b": safe_int(meta.get("max_edges_per_node")),
            "gamma": safe_float(meta.get("gamma"), 1.0),
        },
        "solution_accepted": False,
        "step4_done_hash": "",
        "step5_done_hash": "",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_state(state)
    return state


def active_solution() -> Tuple[Optional[Path], Optional[Dict[str, Any]], Dict[str, Any]]:
    state = load_state()
    raw = state.get("active_solution_path", "")
    if not raw or not Path(raw).exists():
        return None, None, state
    p = Path(raw)
    try:
        return p, load_solution_json(p), state
    except Exception:
        return p, None, state


def solution_binding(json_path: Path) -> Dict[str, Any]:
    data = load_solution_json(json_path)
    meta = data.get("meta", {})
    return {
        "final_json_path": str(json_path.resolve()),
        "final_json_sha256": sha256_file(json_path),
        "k": safe_int(meta.get("k_select")),
        "b": safe_int(meta.get("max_edges_per_node")),
        "gamma": safe_float(meta.get("gamma"), 1.0),
        "selected_count": len(data.get("selected_pairs", []) or []),
        "step1_fingerprint": (
            (read_json(MANIFEST_DIR / "step1_manifest.json", {}) or {}).get("fingerprint", "")
            or (fingerprint(step1_fingerprint(data_dir, review_dir, patent_file))
                if "data_dir" in globals() and "review_dir" in globals() and "patent_file" in globals() else "")
        ),
        "step1_result_sha256": step1_result_sha256(),
        "step2_script_sha256": sha256_file(STEP2_SCRIPT) if STEP2_SCRIPT.exists() else "",
        "step3_script_sha256": sha256_file(STEP3_SCRIPT) if STEP3_SCRIPT.exists() else "",
        "step4_script_sha256": sha256_file(STEP4_SCRIPT) if STEP4_SCRIPT.exists() else "",
        "step5_script_sha256": sha256_file(STEP5_SCRIPT) if STEP5_SCRIPT.exists() else "",
    }


def get_step4_status(json_path: Optional[Path]) -> Dict[str, Any]:
    if json_path is None or not json_path.exists():
        return {"status": "blocked", "label": "尚未确认最终方案"}
    binding = solution_binding(json_path)
    state = load_state()
    candidates = []
    saved = state.get("step4_out_dir", "")
    if saved:
        candidates.append(Path(saved))
    candidates.append(STEP4_OUT)
    seen = set()
    for out_dir in candidates:
        if str(out_dir) in seen:
            continue
        seen.add(str(out_dir))
        manifest = read_json(out_dir / "step6_binding.json", {}) or {}
        if (out_dir / "offline_output_manifest.json").exists() and manifest and manifest.get("final_json_sha256") == binding["final_json_sha256"] and manifest.get("step4_script_sha256") == binding["step4_script_sha256"] and manifest.get("step2_script_sha256") == binding.get("step2_script_sha256", "") and manifest.get("step1_fingerprint", "") == binding.get("step1_fingerprint", "") and manifest.get("step1_result_sha256", "") == binding.get("step1_result_sha256", ""):
            return {"status": "verified", "label": "已有结果 · 与当前方案匹配", "out_dir": str(out_dir)}
        offline = read_json(out_dir / "offline_output_manifest.json", {}) or {}
        if offline:
            fs = offline.get("final_solution", {}) or {}
            if safe_int(fs.get("k_select"), -1) == binding["k"] and safe_int(fs.get("max_edges_per_node"), -1) == binding["b"] and abs(safe_float(fs.get("gamma"), -99) - binding["gamma"]) < 1e-8:
                return {"status": "legacy", "label": "已有案例结果 · 参数匹配，输入指纹待核验", "out_dir": str(out_dir)}
        if (out_dir / "figures").exists() or (out_dir / "tables").exists():
            return {"status": "stale", "label": "发现历史结果 · 与当前方案未能确认匹配", "out_dir": str(out_dir)}
    return {"status": "missing", "label": "尚无当前方案的Step4结果"}


def _completed_step5_sweep(out_dir: Path) -> bool:
    """Require a complete no-LLM control and 5/5 completed repeats per observed tau.

    Completion is a computation status, not proof that the experiment matches
    the current active optimization input. Binding is verified separately.
    """
    report = read_json(out_dir / "llm_full_sweep_summary.json", {}) or {}
    if not isinstance(report, dict):
        return False
    if report.get("status") != "COMPLETE" or report.get("all_complete") is not True:
        return False
    scenarios = report.get("scenarios") or []
    controls = [x for x in scenarios if x.get("scenario") == "no_llm_ablation"]
    temps = [x for x in scenarios if x.get("scenario") == "llm_perturb"]
    return (len(controls) == 1 and controls[0].get("status") == "complete"
            and {round(float(x.get("tau")), 2) for x in temps} == {0.1, 0.3, 0.5}
            and len(temps) == 3 and all(x.get("status") == "complete"
            and int(x.get("complete_repeats") or 0) == 5
            and int(x.get("total_repeats") or 0) == 5 for x in temps))


def get_step5_status(json_path: Optional[Path]) -> Dict[str, Any]:
    if json_path is None or not json_path.exists():
        return {"status": "blocked", "label": "尚未选择方案"}
    binding = solution_binding(json_path)
    state = load_state()
    # Prioritize the latest completed repaired sweep in the research display.
    candidates = [STEP5_REPAIRED_OUT]
    saved = state.get("step5_out_dir", "")
    if saved:
        candidates.append(Path(saved))
    candidates.append(STEP5_OUT)
    seen = set()
    for out_dir in candidates:
        if str(out_dir) in seen or not out_dir.exists():
            continue
        seen.add(str(out_dir))
        manifest = read_json(out_dir / "step6_binding.json", {}) or {}
        matches = (manifest and manifest.get("final_json_sha256") == binding["final_json_sha256"]
            and manifest.get("step5_script_sha256") == binding["step5_script_sha256"]
            and manifest.get("step2_script_sha256") == binding.get("step2_script_sha256", "")
            and manifest.get("step1_fingerprint", "") == binding.get("step1_fingerprint", "")
            and manifest.get("step1_result_sha256", "") == binding.get("step1_result_sha256", ""))
        if _completed_step5_sweep(out_dir):
            return {"status": "verified" if matches else "legacy",
                    "label": "完整稳健性实验 · 当前方案已核验" if matches else "已有完整稳健性实验 · 方案版本待核验",
                    "out_dir": str(out_dir), "all_complete": True}
        # Unfinished directories are NOT treated as valid completed experiments.
    return {"status": "missing", "label": "尚无已核验完成的稳健性实验"}


def choose_output_dir(base: Path, expected_binding: Dict[str, Any], stage: str) -> Path:
    if not base.exists() or not any(base.iterdir()):
        return base
    existing = read_json(base / "step6_binding.json", {}) or {}
    keys = ["final_json_sha256", "step1_fingerprint", "step1_result_sha256", "step2_script_sha256"]
    keys.append("step4_script_sha256" if stage == "step4" else "step5_script_sha256")
    if existing and all(existing.get(k, "") == expected_binding.get(k, "") for k in keys):
        return base
    return base / "runs" / now_id("decision")


def proxy_preflight(proxy: str) -> Tuple[bool, str]:
    """Validate the local HTTP proxy endpoint; never silently use direct network."""
    import socket
    from urllib.parse import urlsplit
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        return False, "代理地址不完整，必须使用含端口的显式HTTP代理。"
    try:
        sock = socket.create_connection((parsed.hostname, parsed.port), timeout=3)
        sock.close()
        return True, "代理端口可达"
    except OSError as exc:
        return False, f"无法连接LLM本地代理 {proxy}：{exc}"


def real_key(key: str) -> bool:
    key = (key or "").strip()
    if not key or "***" in key or key.lower() in {"none", "null", "your_api_key", "sk-xxx"}:
        return False
    return True


def run_step1(data_dir: Path, review_dir: Path, patent_file: Path, proxy: str, api_key: str, base_url: str, model: str, python_exe: str) -> Tuple[bool, str]:
    if not review_dir.exists():
        return False, f"未找到评论数据目录：{review_dir}"
    if not patent_file.exists():
        return False, f"未找到专利文件：{patent_file}"
    if not real_key(api_key):
        return False, "从页面重新执行Step1需要真实LLM_API_KEY。若已有Step1结果，请选择直接加载，避免重复运行。"
    proxy_ok, proxy_message = proxy_preflight(proxy)
    if not proxy_ok:
        return False, proxy_message
    runtime_step1 = prepare_proxy_step1()
    backup = None
    if STEP1_OUT.exists() and any(STEP1_OUT.iterdir()):
        backup = WORKSPACE_DIR / "backups" / now_id("step1_before_rerun")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(STEP1_OUT, backup)
    env = build_env(data_dir, proxy, api_key, base_url, model)
    env.update({
        "STEP1_LLM_REUSE_POLICY": "auto" if api_key else "reuse",
        "STEP1_REVIEW_INPUT_DIR": str(review_dir.resolve()),
        "STEP1_PATENT_XLSX": str(patent_file.resolve()),
        "LLM_RESUME_MODE": "normal",
    })
    code, output, log_path = run_process([python_exe, str(runtime_step1)], env, SCRIPT_DIR, "step1")
    if code == 0 and all(p.exists() for p in STEP1_REQUIRED):
        fp = fingerprint(step1_fingerprint(data_dir, review_dir, patent_file))
        write_json(MANIFEST_DIR / "step1_manifest.json", {
            "fingerprint": fp,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": str(STEP1_SCRIPT),
            "output_dir": str(STEP1_OUT),
            "proxy": proxy,
            "verification": "produced_by_step6",
            "llm_cache_reuse": not bool(api_key),
        })
        return True, f"Step1完成。日志：{log_path}"
    # If the new run failed, restore the previous complete result set.
    if backup is not None and backup.exists():
        if STEP1_OUT.exists():
            shutil.rmtree(STEP1_OUT)
        shutil.copytree(backup, STEP1_OUT)
    return False, f"Step1运行失败或输出不完整；原结果已保留（如原本存在）。退出码={code}；日志：{log_path}\n{output[-3000:]}"


def run_fixed_step3(k: int, b: int, gamma: float, data_dir: Path, review_dir: Path, patent_summary: Path, proxy: str, api_key: str, base_url: str, model: str, python_exe: str) -> Tuple[bool, Optional[Path], Optional[Path], str]:
    helper = prepare_fixed_step3()
    run_dir = STEP3_OUT / "step6_runs" / now_id(f"K{k}_B{b}_G{str(gamma).replace('.', 'p')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(data_dir, proxy, api_key, base_url, model)
    env.update({
        "AGENT_WORK_DIR": str(SCRIPT_DIR),
        "AGENT_OUT_DIR": str(run_dir),
        "AGENT_BATCH_RESULTS_CSV": str(review_dir.resolve()),
        "AGENT_PATENT_SUMMARY_CSV": str(patent_summary.resolve()),
        "STEP1_SCRIPT": str(STEP1_SCRIPT),
        "STEP2_SCRIPT": str(STEP2_SCRIPT),
        "AGENT_ALLOWED_B_VALUES": str(int(b)),
        "AGENT_INITIAL_B": str(int(b)),
        "AGENT_INITIAL_K": str(int(k)),
        "AGENT_MIN_K_FOR_AUTO_SELECT": str(int(k)),
        "AGENT_GAMMA": str(float(gamma)),
        "AGENT_FIXED_KB_ONLY": "1",
        "AGENT_RUN_STEP1": "0",
        "AGENT_USE_LLM_PLANNER": "0",
        "AGENT_DISABLE_STEP2_LLM_PLAN": "1",
        "AGENT_EXACT_CAND_POOL": "0",
        "AGENT_TIME_LIMIT_SEC": "0",
        "AGENT_SELECTION_POLICY": "human_specified_k_b",
    })
    code, output, log_path = run_process([python_exe, str(helper)], env, SCRIPT_DIR, "step23")
    summary_path = run_dir / "agent_run_summary.json"
    if code == 0 and summary_path.exists():
        summary = read_json(summary_path, {}) or {}
        final_json = resolve_recorded_json(str(summary.get("final_json_path", "")))
        if final_json and final_json.exists():
            return True, final_json, summary_path, f"Step2/3完成。日志：{log_path}"
    return False, None, None, f"Step2/3运行失败。退出码={code}；日志：{log_path}\n{output[-4000:]}"


def run_step4(json_path: Path, summary_path: Path, data_dir: Path, review_dir: Path, patent_summary: Path, python_exe: str, proxy: str, api_key: str, base_url: str, model: str) -> Tuple[bool, Path, str]:
    binding = solution_binding(json_path)
    out_dir = choose_output_dir(STEP4_OUT, binding, "step4")
    env = build_env(data_dir, proxy, api_key, base_url, model)
    env.update({"AGENT_USE_LLM_PLANNER": "0", "DISABLE_STEP2_LLM_PLAN": "1", "USE_LLM_FOR_PLAN": "0"})
    cmd = [
        python_exe, str(STEP4_SCRIPT),
        "--work-dir", str(SCRIPT_DIR),
        "--agent-summary", str(summary_path.resolve()),
        "--final-json", str(json_path.resolve()),
        "--final-csv", str(json_path.with_suffix(".csv").resolve()),
        "--step2-script", str(STEP2_SCRIPT.resolve()),
        "--review-input", str(review_dir.resolve()),
        "--patent-summary", str(patent_summary.resolve()),
        "--out-dir", str(out_dir.resolve()),
        "--bootstrap-n", "0",
    ]
    code, output, log_path = run_process(cmd, env, SCRIPT_DIR, "step4")
    if code == 0 and (out_dir / "offline_output_manifest.json").exists():
        write_json(out_dir / "step6_binding.json", {**binding, "created_at": datetime.now().isoformat(timespec="seconds")})
        return True, out_dir, f"Step4完成。日志：{log_path}"
    return False, out_dir, f"Step4运行失败。退出码={code}；日志：{log_path}\n{output[-4000:]}"


def run_step5(json_path: Path, summary_path: Path, data_dir: Path, review_dir: Path, patent_file: Path, proxy_step1: Path, api_key: str, base_url: str, model: str, proxy: str, python_exe: str, repeats: int = 5) -> Tuple[bool, Path, str]:
    if not real_key(api_key):
        return False, STEP5_OUT, "Step5需要真实LLM_API_KEY，请先在研究者设置中填写。"
    proxy_ok, proxy_message = proxy_preflight(proxy)
    if not proxy_ok:
        return False, STEP5_OUT, proxy_message
    binding = solution_binding(json_path)
    out_dir = choose_output_dir(STEP5_OUT, binding, "step5")
    env = build_env(data_dir, proxy, api_key, base_url, model)
    cmd = [
        python_exe, str(STEP5_SCRIPT),
        "--work-dir", str(SCRIPT_DIR),
        "--step1-script", str(proxy_step1.resolve()),
        "--step2-script", str(STEP2_SCRIPT.resolve()),
        "--agent-summary", str(summary_path.resolve()),
        "--final-json", str(json_path.resolve()),
        "--step1-dir", str(STEP1_OUT.resolve()),
        "--batch-input", str(review_dir.resolve()),
        "--out-dir", str(out_dir.resolve()),
        "--patent-file", str(patent_file.resolve()),
        "--scope", "all",
        "--n-repeats", str(int(repeats)),
        "--analysis-mode", "tau_sweep",
        "--tau-values", "0.1", "0.3", "0.5",
        "--gamma", str(binding["gamma"]),
        "--k-select", str(binding["k"]),
        "--max-edges-per-node", str(binding["b"]),
        "--exact-cand-pool", "0",
        "--llm-base-url", base_url,
        "--llm-model", model,
    ]
    code, output, log_path = run_process(cmd, env, SCRIPT_DIR, "step5")
    required = [out_dir / "tables" / "llm_tau_sweep_runs_index.csv", out_dir / "tables" / "llm_analysis_summary_by_scenario.csv"]
    if code == 0 and all(p.exists() for p in required):
        write_json(out_dir / "step6_binding.json", {**binding, "created_at": datetime.now().isoformat(timespec="seconds"), "n_repeats": repeats, "tau_values": [0.1, 0.3, 0.5], "scope": "all"})
        return True, out_dir, f"Step5完成。日志：{log_path}"
    return False, out_dir, f"Step5运行失败或输出不完整。退出码={code}；日志：{log_path}\n{output[-4000:]}"


def get_current_candidate_and_selected(json_path: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate = build_candidate_edges()
    selected = pd.DataFrame()
    if json_path and json_path.exists():
        try:
            selected = pd.DataFrame(load_solution_json(json_path).get("selected_pairs", []) or [])
        except Exception:
            pass
    return candidate, selected


def _plotly_base(fig, height: int = 420):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=12, color="#243B53"),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=24, r=24, t=22, b=42),
        hoverlabel=dict(bgcolor="#102A43", font=dict(color="#FFFFFF", size=12), bordercolor="#102A43"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.78)", bordercolor="#E6EDF3", borderwidth=1,
            font=dict(size=11, color="#526579"),
        ),
    )
    fig.update_xaxes(showline=False, zeroline=False, gridcolor="#EEF2F6", tickfont=dict(color="#66788A"))
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="#EEF2F6", tickfont=dict(color="#66788A"))
    return fig


def paper_review_network_figure(selected: pd.DataFrame, height: int = 365):
    """Render the selected collaboration network with the same layout and encoding as Figure 2."""
    if go is None or selected is None or selected.empty:
        return None
    data = selected.copy()
    if "combined_weight" not in data.columns:
        data["combined_weight"] = 0.0
    data["combined_weight"] = pd.to_numeric(data["combined_weight"], errors="coerce").fillna(0.0)
    data = data.sort_values("combined_weight", ascending=True, kind="mergesort")

    if nx is not None:
        graph = nx.Graph()
        for _, row in data.iterrows():
            a, b = str(row.get("product_a", "")).strip(), str(row.get("product_b", "")).strip()
            if a and b and a != b:
                weight = safe_float(row.get("combined_weight"))
                graph.add_edge(a, b, metric=weight, weight=weight)
        if graph.number_of_edges() == 0:
            return None
        positions = nx.kamada_kawai_layout(graph, weight="weight")
        positions = {name: (xy[0] * 1.15, xy[1] * 1.15) for name, xy in positions.items()}
    else:
        import math
        nodes = sorted(set(data["product_a"].astype(str)) | set(data["product_b"].astype(str)))
        graph = None
        positions = {
            name: (math.cos(2 * math.pi * i / max(1, len(nodes))),
                   math.sin(2 * math.pi * i / max(1, len(nodes))))
            for i, name in enumerate(nodes)
        }

    weighted_degree = {}
    for node in positions:
        if graph is not None:
            weighted_degree[node] = sum(safe_float(graph[node][nb].get("metric")) for nb in graph.neighbors(node))
        else:
            weighted_degree[node] = 0.0
    max_wdeg = max(weighted_degree.values()) if weighted_degree else 1.0
    max_edge = max((safe_float(d.get("metric")) for _, _, d in graph.edges(data=True)), default=1.0) if graph is not None else 1.0
    core_nodes = set(sorted(positions, key=lambda name: weighted_degree.get(name, 0.0), reverse=True)[:4])

    fig = go.Figure()
    edge_x, edge_y, edge_labels, edge_hover = [], [], [], []
    edges = list(graph.edges(data=True)) if graph is not None else []
    edges.sort(key=lambda item: safe_float(item[2].get("metric")))
    for a, b, attrs in edges:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        weight = safe_float(attrs.get("metric"))
        norm = weight / max_edge if max_edge > 0 else 0.0
        alpha = 0.25 + 0.60 * norm
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=0.8 + 5.0 * norm, color=f"rgba(88,97,116,{alpha:.3f})"),
            hoverinfo="text",
            text=[f"{paper_product_label(a)}—{paper_product_label(b)}：{weight:.4f}"] * 2,
            showlegend=False, name="候选协同边",
        ))
        edge_x.append((x0 + x1) / 2.0)
        edge_y.append((y0 + y1) / 2.0)
        edge_labels.append(f"{weight:.2f}")
    if edge_labels:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="text", text=edge_labels,
            textfont=dict(size=9, color="#374151", family="Arial, Microsoft YaHei, sans-serif"),
            hoverinfo="skip", showlegend=False, name="综合权重",
        ))

    for group_nodes, fill, edge, legend in [
        ([n for n in positions if n in core_nodes], "#F4A261", "#9A5A00", "核心候选节点"),
        ([n for n in positions if n not in core_nodes], "#8ECAE6", "#2B6F8A", "普通候选节点"),
    ]:
        if not group_nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[positions[n][0] for n in group_nodes],
            y=[positions[n][1] for n in group_nodes],
            mode="markers+text",
            text=[str(n) for n in group_nodes],
            textposition="middle center",
            textfont=dict(size=10, color="#1F2937", family="Arial, Microsoft YaHei, sans-serif"),
            marker=dict(
                size=[23 + 18 * weighted_degree.get(n, 0.0) / max(1e-9, max_wdeg) for n in group_nodes],
                color=fill, line=dict(color=edge, width=2), opacity=0.96,
            ),
            hovertext=[
                f"{paper_product_label(n)}<br>加权参与度：{weighted_degree.get(n, 0.0):.3f}"
                for n in group_nodes
            ],
            hoverinfo="text", name=legend, showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="lines",
        line=dict(color="#586174", width=3), name="候选协同边", showlegend=True,
    ))
    fig.update_layout(
        template="plotly_white", height=height, autosize=True,
        margin=dict(l=12, r=12, t=10, b=42),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, Microsoft YaHei, sans-serif", color="#1F2937", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="left", x=0,
                    font=dict(size=9, color="#475467"), bgcolor="rgba(255,255,255,.88)"),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )
    return fig


def network_figure(candidate: pd.DataFrame, selected: pd.DataFrame, mode: str = "筛选对照", paper_style: bool = False, height: int = 590):
    if paper_style:
        return paper_review_network_figure(selected, height=height)
    if go is None or candidate.empty:
        return None
    nodes = sorted(set(candidate["product_a"].astype(str)) | set(candidate["product_b"].astype(str)))
    G = nx.Graph() if nx is not None else None
    if G is not None:
        G.add_nodes_from(nodes)
        for _, r in candidate.iterrows():
            G.add_edge(str(r.product_a), str(r.product_b), weight=safe_float(r.get("pair_freq"), 1.0))
        pos = nx.spring_layout(G, seed=26, k=0.9, iterations=140, weight="weight")
    else:
        import math
        pos = {n: (math.cos(2 * math.pi * i / max(1, len(nodes))), math.sin(2 * math.pi * i / max(1, len(nodes)))) for i, n in enumerate(nodes)}

    selected_keys = set()
    selected_degree: Dict[str, int] = {n: 0 for n in nodes}
    for _, r in selected.iterrows():
        a, b = str(r.get("product_a", "")), str(r.get("product_b", ""))
        selected_keys.add("|".join(pair_key(a, b)))
        selected_degree[a] = selected_degree.get(a, 0) + 1
        selected_degree[b] = selected_degree.get(b, 0) + 1

    fig = go.Figure()
    # Non-selected candidate links remain visible as a muted gray background
    # in the final network; selected links are drawn above them in blue.
    cx, cy = [], []
    for _, r in candidate.iterrows():
        a, b = str(r.product_a), str(r.product_b)
        edge_key = "|".join(pair_key(a, b))
        if mode == "优化网络" and edge_key in selected_keys:
            continue
        cx.extend([pos[a][0], pos[b][0], None])
        cy.extend([pos[a][1], pos[b][1], None])
    if cx:
        background_color = "rgba(130,145,160,0.72)" if mode == "优化网络" else "rgba(151,166,181,0.30)"
        background_width = 1.7 if mode == "优化网络" else 1.15
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color=background_color, width=background_width),
            hoverinfo="skip",
            name=("未入选候选关系" if mode == "优化网络" else "候选关系"),
            showlegend=(mode not in {"候选网络", "优化网络"}),
        ))

    # Selected links: line width reflects relative combined weight.
    edge_label_x, edge_label_y, edge_label_text = [], [], []
    if mode != "候选网络" and not selected.empty:
        vals = [safe_float(r.get("combined_weight")) for _, r in selected.iterrows()]
        vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
        for _, r in selected.iterrows():
            a, b = str(r.get("product_a", "")), str(r.get("product_b", ""))
            if a not in pos or b not in pos:
                continue
            w = safe_float(r.get("combined_weight"))
            norm = 0.5 if vmax == vmin else (w - vmin) / (vmax - vmin)
            hover = (
                f"<b>{html_lib.escape(a)} — {html_lib.escape(b)}</b>"
                f"<br>综合权重：{w:.4f}"
                f"<br>需求强度：{safe_float(r.get('demand_norm')):.3f}"
                f"<br>技术强度：{safe_float(r.get('tech_norm')):.3f}"
            )
            fig.add_trace(go.Scatter(
                x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                line=dict(color=("#7C8797" if paper_style else "#2563EB"), width=1.8 + 3.0 * norm),
                opacity=0.92, hoverinfo="text", text=[hover, hover],
                showlegend=False, name="入选关系",
            ))
            if paper_style:
                edge_label_x.append((pos[a][0] + pos[b][0]) / 2.0)
                edge_label_y.append((pos[a][1] + pos[b][1]) / 2.0)
                edge_label_text.append(f"{w:.2f}")
    if paper_style and edge_label_text:
        fig.add_trace(go.Scatter(
            x=edge_label_x, y=edge_label_y, mode="text", text=edge_label_text,
            textfont=dict(size=8, color="#667487", family="Arial, Microsoft YaHei, sans-serif"),
            hoverinfo="skip", showlegend=False, name="综合权重",
        ))

    candidate_degree: Dict[str, int] = {n: 0 for n in nodes}
    for _, r in candidate.iterrows():
        candidate_degree[str(r.product_a)] = candidate_degree.get(str(r.product_a), 0) + 1
        candidate_degree[str(r.product_b)] = candidate_degree.get(str(r.product_b), 0) + 1
    if paper_style:
        node_colors = ["#F2A24A" if selected_degree.get(n, 0) > 0 else "#78BBD2" for n in nodes]
        node_sizes = [20 + 2.2 * selected_degree.get(n, 0) for n in nodes]
        node_line = ["#D77F28" if selected_degree.get(n, 0) > 0 else "#D5EAF1" for n in nodes]
    else:
        node_colors = ["#2563EB" if selected_degree.get(n, 0) > 0 else "#C5CFD8" for n in nodes]
        node_sizes = [23 + 3.2 * selected_degree.get(n, 0) for n in nodes]
        node_line = ["#D5F0EA" if selected_degree.get(n, 0) > 0 else "#E8EDF2" for n in nodes]
    hover = [
        f"<b>{html_lib.escape(n)}</b><br>候选连接：{candidate_degree.get(n, 0)}<br>入选连接：{selected_degree.get(n, 0)}<br>状态：{'已入选' if selected_degree.get(n, 0) else '未入选'}"
        for n in nodes
    ]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        textfont=dict(size=(10 if paper_style else 12), color="#30465B", family="Arial, Microsoft YaHei, sans-serif"),
        marker=dict(size=node_sizes, color=node_colors, line=dict(color=node_line, width=3), opacity=0.98),
        hovertext=hover, hoverinfo="text", name="产品节点", showlegend=False,
    ))
    _plotly_base(fig, height=height)
    fig.update_layout(
        margin=dict(l=18, r=18, t=12, b=18), showlegend=False,
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        hovermode="closest", dragmode="pan", uirevision=f"network-{mode}",
    )
    return fig


def parameter_scatter(cache: List[Dict[str, Any]]):
    if go is None or not cache:
        return None
    df = pd.DataFrame([x for x in cache if x.get("input_status") != "stale"]).drop_duplicates(subset=["k", "b", "gamma", "path"])
    if df.empty:
        return None
    df = df[df["gamma"].between(-100, 100)]
    fig = px.scatter(
        df, x="k", y="weight", color="b", symbol="gamma", size="selected_count",
        size_max=18, hover_data={"gamma": ":.2f", "selected_count": True, "optimal": True, "source": True},
        labels={"k": "K · 产品对数量", "weight": "目标函数值", "b": "B · 最大节点度", "gamma": "γ"},
        template="plotly_white", color_continuous_scale=["#C9D9E5", "#2563EB", "#164E63"],
    )
    fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=1.5), opacity=0.9))
    _plotly_base(fig, height=390)
    fig.update_layout(coloraxis_colorbar=dict(title="B", thickness=10, len=0.72), legend_title_text="γ")
    return fig


def gamma_figure(cache: List[Dict[str, Any]], k: int, b: int):
    if go is None:
        return None
    rows = [x for x in cache if x["k"] == int(k) and x["b"] == int(b)]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("gamma").drop_duplicates("gamma")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.gamma, y=df.weight, mode="lines+markers",
        line=dict(color="#2563EB", width=3, shape="linear"),
        marker=dict(size=10, color="#2563EB", line=dict(color="#FFFFFF", width=2)),
        name="总目标值", customdata=df[["selected_count", "optimal"]].to_numpy(),
        hovertemplate="γ=%{x:.2f}<br>目标值=%{y:.4f}<br>入选产品对=%{customdata[0]}<br>最优性=%{customdata[1]}<extra></extra>",
    ))
    _plotly_base(fig, height=340)
    fig.update_layout(
        showlegend=False, hovermode="x unified",
        xaxis_title="γ · 技术增益系数", yaxis_title="总目标值",
        margin=dict(l=30, r=20, t=12, b=48),
    )
    return fig


# -----------------------------------------------------------------------------
# Completed gamma sensitivity: local, read-only historical experiment evidence
# -----------------------------------------------------------------------------
def _gamma_edge_keys(solution: Dict[str, Any]) -> set:
    """Unordered product-pair IDs. Never compare by row position or weight."""
    edges = set()
    for item in (solution.get("selected_pairs") or []):
        if not isinstance(item, dict):
            continue
        a, b = str(item.get("product_a", "")).strip(), str(item.get("product_b", "")).strip()
        if a and b and a != b:
            edges.add("||".join(sorted((a, b))))
    return edges


def _gamma_edge_text(key: str) -> str:
    return "—".join(str(key).split("||", 1))


def _load_one_gamma_run(run_dir: Path) -> Dict[str, Any]:
    """Verify one recorded sweep using its ORIGINAL local JSON/CSV results.

    Absolute paths embedded in the historical CSV came from the experiment's
    Windows computer. Do not use those paths: reconstruct paths relative to the
    selected run directory. Missing/invalid results are never reported as done.
    """
    summary_file = run_dir / "gamma_sensitivity_summary.csv"
    manifest = read_json(run_dir / "gamma_run_manifest.json", {}) or {}
    raw = read_csv_any(summary_file)
    rows, issues = [], []
    if not {"k", "b", "gamma", "status", "total_weight"}.issubset(raw.columns):
        raise ValueError("敏感性汇总CSV缺少必需字段")
    base_gamma = safe_float(manifest.get("baseline_gamma", 1.0), 1.0)
    expected = manifest.get("gamma_values", [])
    for _, r in raw.iterrows():
        try:
            k = int(r["k"])
            b = int(r["b"])
            gamma = float(r["gamma"])
            if str(r.get("status", "")).strip().lower() != "success" or safe_int(r.get("return_code", 0), -1) != 0:
                raise ValueError("该情景运行未成功")
            token = str(float(gamma)).replace(".", "p")
            json_file = run_dir / ("gamma_" + token) / "edgeset_selection_exact.json"
            if not json_file.is_file():
                raise FileNotFoundError("缺少本地精确优化JSON")
            sol = load_solution_json(json_file)
            meta, solver = sol.get("meta", {}), (sol.get("meta", {}).get("solver") or {})
            if (safe_int(meta.get("k_select"), -1) != k or
                safe_int(meta.get("max_edges_per_node"), -1) != b or
                abs(safe_float(meta.get("gamma", -999), -999) - gamma) > 1e-7):
                raise ValueError("CSV与优化JSON的K/B/γ不匹配")
            if not solver.get("optimal", False):
                raise ValueError("该情景未被确认精确最优")
            weight = safe_float(solver.get("best_weight"), float("nan"))
            summary_weight = safe_float(r.get("total_weight"), float("nan"))
            if pd.isna(weight) or pd.isna(summary_weight) or abs(weight-summary_weight) > 1e-5 * max(1.0, abs(weight)):
                raise ValueError("汇总目标值与优化JSON不一致")
            edges = _gamma_edge_keys(sol)
            n_selected = safe_int(r.get("selected_count"), -1)
            if len(edges) != len(sol.get("selected_pairs") or []) or len(edges) != n_selected:
                raise ValueError("入选产品对数量或唯一标识不一致")
            rows.append({
                "k": k, "b": b, "gamma": gamma, "objective": weight,
                "avg_edge_weight": weight / max(1, len(edges)),
                "selected_count": len(edges), "unique_nodes": safe_int(r.get("unique_nodes"), 0),
                "pairs": edges, "solution_path": json_file,
                "solution": sol,
            })
        except Exception as exc:
            issues.append("γ=%s：%s" % (r.get("gamma", "?"), exc))

    baseline = next((x for x in rows if abs(x["gamma"] - base_gamma) < 1e-7), None)
    if baseline is None:
        issues.append("缺少基准γ=%.2f的有效优化结果，不能计算与基准方案的差异" % base_gamma)
    for rec in rows:
        if baseline is not None and rec["k"] == baseline["k"] and rec["b"] == baseline["b"]:
            baseline_edges = baseline["pairs"]
            rec["overlap"] = len(rec["pairs"] & baseline_edges)
            rec["overlap_rate"] = len(rec["pairs"] & baseline_edges) / max(1, len(baseline_edges))
            rec["added"] = sorted(rec["pairs"] - baseline_edges)
            rec["removed"] = sorted(baseline_edges - rec["pairs"])
        else:
            rec.update({"overlap": None, "overlap_rate": None, "added": [], "removed": []})
    rows.sort(key=lambda r: (r["k"], r["b"], r["gamma"]))
    expected_count = len(expected) if isinstance(expected, list) and expected else len(raw)
    manifest_match = True
    if isinstance(expected, list) and expected:
        observed = {round(x["gamma"], 8) for x in rows}
        manifest_match = all(round(float(g), 8) in observed for g in expected)
    return {
        "run_dir": run_dir, "run_id": str(manifest.get("run_id", run_dir.name)),
        "base_gamma": base_gamma, "baseline": baseline, "rows": rows,
        "issues": issues, "expected_count": expected_count,
        "complete": len(rows) == len(raw) == expected_count and not issues and manifest_match,
        "mtime": summary_file.stat().st_mtime,
    }


def find_gamma_sensitivity_run(k: Optional[int] = None, b: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Prefer a complete sweep, then the most recent relevant sweep."""
    if not GAMMA_OUT.is_dir():
        return None
    candidates = []
    for csv_file in GAMMA_OUT.glob("*/gamma_sensitivity_summary.csv"):
        try:
            info = _load_one_gamma_run(csv_file.parent)
            if not info["rows"]:
                continue
            if k is not None and b is not None and not any(x["k"] == int(k) and x["b"] == int(b) for x in info["rows"]):
                continue
            candidates.append(info)
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda d: (bool(d["complete"]), d["mtime"]))


def _gamma_display_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    view = []
    for r in rows:
        view.append({
            "γ": round(r["gamma"], 2),
            "目标函数值": round(r["objective"], 4),
            "平均边权重": round(r["avg_edge_weight"], 4),
            "入选产品对": r["selected_count"],
            "覆盖产品": r["unique_nodes"],
            "基准重合数": r["overlap"] if r["overlap"] is not None else "—",
            "基准重合率": ("%.2f%%" % (100 * r["overlap_rate"])) if r["overlap_rate"] is not None else "—",
            "相对基准新增": len(r["added"]),
            "相对基准退出": len(r["removed"]),
        })
    return pd.DataFrame(view)


def gamma_sensitivity_charts(rows: List[Dict[str, Any]], baseline_gamma: float):
    if go is None or not rows:
        return None, None
    x = [r["gamma"] for r in rows]
    obj = go.Figure()
    obj.add_trace(go.Scatter(x=x, y=[r["objective"] for r in rows], mode="lines+markers",
                             line=dict(color="#2563EB", width=3), marker=dict(size=9), name="目标函数值"))
    obj.add_trace(go.Scatter(x=[baseline_gamma, baseline_gamma],
                             y=[min(r["objective"] for r in rows), max(r["objective"] for r in rows)],
                             mode="lines", line=dict(color="#9AAABA", dash="dash", width=1.5),
                             name="基准γ", hoverinfo="skip"))
    _plotly_base(obj, height=320)
    obj.update_layout(xaxis_title="γ · 技术增益系数", yaxis_title="目标函数值",
                      hovermode="x unified", margin=dict(l=30, r=20, t=10, b=40))
    overlaps = [r for r in rows if r["overlap_rate"] is not None]
    overlap_fig = None
    if overlaps:
        overlap_fig = go.Figure()
        overlap_fig.add_trace(go.Scatter(x=[r["gamma"] for r in overlaps],
                                         y=[100 * r["overlap_rate"] for r in overlaps],
                                         mode="lines+markers", line=dict(color="#2C6591", width=3),
                                         marker=dict(size=9), name="产品对重合率"))
        _plotly_base(overlap_fig, height=320)
        overlap_fig.update_layout(xaxis_title="γ · 技术增益系数", yaxis_title="相对基准方案重合率（%）",
                                  yaxis=dict(range=[0, 105]), hovermode="x unified",
                                  margin=dict(l=30, r=20, t=10, b=40))
    return obj, overlap_fig


def display_gamma_sensitivity_panel(key_prefix: str = "parameter", k: Optional[int] = None,
                                    b: Optional[int] = None, compact: bool = False) -> None:
    """Read-only interpretation of completed experiments; never triggers Step2."""
    st.markdown("**技术增益系数 γ 的敏感性分析**")
    info = find_gamma_sensitivity_run(k, b)
    if info is None:
        st.caption("未找到对应的历史γ敏感性实验目录。此处仅展示已有结果，不会自动重新运行实验。")
        return
    rows = [r for r in info["rows"] if (k is None or r["k"] == int(k)) and (b is None or r["b"] == int(b))]
    if not rows:
        st.caption("历史γ实验不包含当前选定的K/B组合。")
        return
    baseline = info["baseline"]
    st.caption("历史实验：K=%d，B=%d；γ=%.2f为比较基准。该分析复用原始精确求解结果，不改变当前交互方案。" %
               (rows[0]["k"], rows[0]["b"], info["base_gamma"]))
    if info["complete"]:
        st.success("已读取%d/%d组成功且与优化JSON一致的γ情景。" % (len(rows), info["expected_count"]))
    else:
        st.warning("只读取到%d/%d组通过校验的γ情景；以下仅展示核验通过的部分结果。" %
                   (len(rows), info["expected_count"]))
    if info["issues"]:
        with st.expander("查看未通过核验的实验记录", expanded=False):
            for item in info["issues"]:
                st.write(item)

    display_rows = _gamma_display_table(rows)
    # Charts show the weighted objective defined for each gamma. Cross-gamma
    # changes in objective value are NOT improvements under a common metric.
    if go is not None:
        fig1, fig2 = gamma_sensitivity_charts(rows, info["base_gamma"])
        c1, c2 = st.columns(2)
        with c1:
            if fig1 is not None:
                st.plotly_chart(fig1, use_container_width=True)
        with c2:
            if fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
    st.caption("目标函数随γ变化同时受到权重定义变化的影响；方案稳定性以相同候选范围内的入选产品对重合及替换情况衡量。")
    dataframe_compat(display_rows, use_container_width=True, height=350 if not compact else 260)
    if compact:
        return
    csv_data = display_rows.to_csv(index=False).encode("utf-8-sig")
    st.download_button("导出γ敏感性汇总", data=csv_data, file_name="gamma_sensitivity_step6_summary.csv",
                       mime="text/csv", key=key_prefix+"_gamma_download")
    differing = [x for x in rows if x["added"] or x["removed"]]
    default = differing[0] if differing else next((x for x in rows if abs(x["gamma"]-info["base_gamma"]) < 1e-7), rows[0])
    options = ["%.2f" % x["gamma"] for x in rows]
    selection = st.selectbox("查看指定γ下的入选产品对变化", options,
                             index=options.index("%.2f" % default["gamma"]), key=key_prefix+"_gamma_detail")
    rec = next(x for x in rows if abs(x["gamma"]-float(selection)) < 1e-8)
    if baseline is None:
        st.info("没有有效基准方案，暂不能计算产品对变化。")
    elif not (rec["added"] or rec["removed"]):
        st.info("γ=%.2f与基准方案入选产品对相同。" % rec["gamma"])
    else:
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("**相对基准新增的产品对（%d条）**" % len(rec["added"]))
            for edge in rec["added"]:
                st.write("+ " + _gamma_edge_text(edge))
        with rcol:
            st.markdown("**相对基准退出的产品对（%d条）**" % len(rec["removed"]))
            for edge in rec["removed"]:
                st.write("− " + _gamma_edge_text(edge))
    with st.expander("查看该γ下的全部入选产品对与需求—技术证据", expanded=False):
        selected = pd.DataFrame(rec["solution"].get("selected_pairs", []))
        if not selected.empty:
            dataframe_compat(decision_explanation_table(selected), use_container_width=True, height=330)

def display_figures(folder: Path, max_items: int = 12) -> None:
    """Display local figures robustly across Streamlit versions.

    Step4 contains some very high-resolution network plots (up to ~9k px).
    Passing a path string directly to st.image can be version-sensitive and
    needlessly sends huge images to the browser. Decode locally, downsample,
    and pass a PIL image; fall back to bytes if Pillow is unavailable.
    """
    if not folder.exists() or not folder.is_dir():
        st.info("当前结果目录没有可展示图片。")
        return
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    if not files:
        st.info("当前目录尚无可展示图片。")
        return
    files = files[:max_items]
    for i in range(0, len(files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(files):
                continue
            path = files[idx]
            with col:
                title = path.stem.replace("_", " ")
                st.markdown(
                    f'<div class="figure-card-title"><span>FIGURE {idx + 1:02d}</span>'
                    f'<b>{html_lib.escape(title)}</b></div>',
                    unsafe_allow_html=True,
                )
                try:
                    if Image is not None:
                        with Image.open(path) as source_image:
                            source_image.load()
                            # Downsample before RGB conversion to limit peak
                            # memory use for dense 9k-by-9k network diagrams.
                            resampling = getattr(Image, "Resampling", Image)
                            source_image.thumbnail((2400, 1800), getattr(resampling, "LANCZOS", 1))
                            image = source_image.convert("RGB")
                            st.image(image, use_column_width=True)
                    else:
                        st.image(path.read_bytes(), use_column_width=True)
                except Exception as exc:
                    st.warning(f"图片读取失败：{path.name}（{exc}）")
                    try:
                        st.download_button(
                            "下载原图", data=path.read_bytes(), file_name=path.name,
                            mime="image/png" if path.suffix.lower() == ".png" else "image/jpeg",
                            key="download_figure_" + hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12],
                        )
                    except Exception:
                        pass


def status_badge(status: str, label: str) -> None:
    colors = {
        "verified": ("#0F766E", "#E8F6F3", "●"),
        "legacy": ("#9A6700", "#FFF7E6", "◷"),
        "stale": ("#B42318", "#FFF0EF", "!"),
        "missing": ("#526579", "#EEF2F6", "○"),
        "blocked": ("#526579", "#EEF2F6", "—"),
    }
    color, bg, icon = colors.get(status, ("#526579", "#EEF2F6", "○"))
    st.markdown(
        f'<span class="status-pill" style="--pill-color:{color};--pill-bg:{bg}"><i>{icon}</i>{html_lib.escape(label)}</span>',
        unsafe_allow_html=True,
    )


def metric_row(items: List[Tuple[str, Any, str]], class_name: str = "metric-grid") -> None:
    cards = []
    for idx, (label, value, delta) in enumerate(items):
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html_lib.escape(str(label))}</div>'
            f'<div class="metric-value">{html_lib.escape(str(value))}</div>'
            f'<div class="metric-note">{html_lib.escape(str(delta or ""))}</div>'
            '</div>'
        )
    safe_class = re.sub(r'[^a-zA-Z0-9_-]', '', str(class_name)) or 'metric-grid'
    st.markdown('<div class="' + safe_class + '">' + ''.join(cards) + '</div>', unsafe_allow_html=True)



def paper_product_label(value: Any) -> str:
    text = str(value).strip()
    return "小米App" if text.lower() == "小米app" else text


def decision_explanation_table(selected: pd.DataFrame) -> pd.DataFrame:
    """Present the original scored product pairs with their actual evidence fields."""
    fields = {
        "product_a": "产品A", "product_b": "产品B", "combined_weight": "综合优先级",
        "demand_norm": "需求强度(归一化)", "tech_norm": "技术支撑(归一化)",
        "scenario_primary": "主要改进类别", "issue_primary": "改进方向", "ipc_primary": "相关IPC小类",
    }
    present = [c for c in fields if c in selected.columns]
    view = selected[present].rename(columns={c: fields[c] for c in present}).copy()
    for col in ("综合优先级", "需求强度(归一化)", "技术支撑(归一化)"):
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(3)
    if "综合优先级" in view.columns:
        view = view.sort_values("综合优先级", ascending=False)
    return view.reset_index(drop=True)

def render_app_hero(page: str, workflow_state: Dict[str, Any], step1_info: Dict[str, Any], active_json_path: Optional[Path], active_json: Optional[Dict[str, Any]]) -> None:
    """Render the same restrained page header used in Figure 2 for every module."""
    page_meta = {
        "证据构建": ("需求与技术证据", "从用户需求与技术储备中识别跨产品协同改进机会。", "EVIDENCE FOUNDATION"),
        "参数与优化": ("协同方案比较", "比较改进价值、方案规模与产品参与约束，选择适合的组合。", "OPTIMIZATION WORKSPACE"),
        "人机评审": ("人机协同评审", "检查产品协同关系及其需求—技术依据，记录管理判断并确认改进组合。", "HUMAN-IN-THE-LOOP"),
        "案例分析": ("协同改进成果", "查看最终入选产品关系、具体改进方向及其需求—技术依据。", "CASE ANALYSIS"),
        "实验验证": ("方案稳健性", "检视技术增益、基线比较与语义扰动下的方案稳定性。", "ROBUSTNESS & VALIDATION"),
    }
    title, desc, eyebrow = page_meta[page]
    # Public mode changes the data source, not the identity of the current module.
    # Keep each module's distinct breadcrumb in both local and public modes.
    if page == "人机评审":
        accepted = bool(
            active_json_path and workflow_state.get("solution_accepted")
            and workflow_state.get("accepted_solution_hash") == sha256_file(active_json_path)
        )
        status_text = "当前方案已确认" if accepted else "当前方案等待评审"
        status_class = "accepted" if accepted else "pending"
    elif page == "证据构建":
        confirmed = bool(workflow_state.get("step1_confirmed"))
        status_text = "证据已确认" if confirmed else "证据待确认"
        status_class = "accepted" if confirmed else "waiting"
    elif active_json_path:
        status_text = "优化方案已加载"
        status_class = "accepted"
    else:
        status_text = "等待优化方案"
        status_class = "waiting"
    st.markdown(
        '<div class="paper-review-header">'
        '<div><div class="paper-review-breadcrumb">RESEARCH WORKSPACE&nbsp;&nbsp; / &nbsp;&nbsp;'
        + html_lib.escape(eyebrow) + '</div>'
        + '<div class="paper-review-title">' + html_lib.escape(title) + '</div>'
        + '<div class="paper-review-subtitle">' + html_lib.escape(desc) + '</div></div>'
        + f'<div class="paper-review-status {status_class}"><span></span>{html_lib.escape(status_text)}</div>'
        + '</div>',
        unsafe_allow_html=True,
    )


def render_human_review_header(workflow_state: Dict[str, Any], active_json_path: Optional[Path]) -> None:
    accepted = bool(
        active_json_path and workflow_state.get("solution_accepted") and
        workflow_state.get("accepted_solution_hash") == sha256_file(active_json_path)
    )
    label = "当前方案已确认" if accepted else "当前方案等待评审"
    cls = "accepted" if accepted else "pending"
    st.markdown(
        '<div class="paper-review-header">'
        '<div class="paper-review-heading">'
        '<div class="paper-review-breadcrumb">RESEARCH WORKSPACE&nbsp;&nbsp; / &nbsp;&nbsp; HUMAN-IN-THE-LOOP</div>'
        '<div class="paper-review-title">人机协同评审</div>'
        '<div class="paper-review-subtitle">检查产品协同关系及其需求—技术依据，记录管理判断并确认改进组合。</div>'
        '</div>'
        f'<div class="paper-review-status {cls}"><span></span>{html_lib.escape(label)}</div>'
        '</div>', unsafe_allow_html=True,
    )


def render_workflow_rail(page: str, workflow_state: Dict[str, Any], active_hash: str) -> None:
    pages = ["证据构建", "参数与优化", "人机评审", "案例分析", "实验验证"]
    labels = ["证据", "优化", "评审", "案例", "验证"]
    done = [
        bool(workflow_state.get("step1_confirmed")),
        bool(workflow_state.get("active_solution_path")),
        bool(workflow_state.get("solution_accepted") and workflow_state.get("accepted_solution_hash") == active_hash and active_hash),
        bool(active_hash and workflow_state.get("step4_done_hash") == active_hash),
        bool(active_hash and workflow_state.get("step5_done_hash") == active_hash),
    ]
    parts = []
    for i, (p, label) in enumerate(zip(pages, labels)):
        cls = "done" if done[i] else ("current" if p == page else "pending")
        icon = "✓" if done[i] else f"0{i+1}"
        parts.append(f'<div class="workflow-step {cls}"><div class="workflow-step-index">{icon}</div><div class="workflow-step-name">{label}</div></div>')
    st.markdown('<div class="workflow-rail">' + ''.join(parts) + '</div>', unsafe_allow_html=True)

def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# -----------------------------------------------------------------------------
# Load saved config/state and render shell
# -----------------------------------------------------------------------------
ensure_dirs()

MODE_ENV = os.getenv("STEP6_MODE", "auto").strip().lower()
with st.sidebar:
    st.markdown("""
    <div class="paper-sidebar-brand">
      <div class="paper-brand-mark">E</div>
      <div><div class="paper-brand-name">Online demonstration</div><div class="paper-brand-sub">platform</div></div>
    </div>
    """, unsafe_allow_html=True)

LOCAL_PIPELINE_AVAILABLE = all(p.exists() for p in (
    STEP1_SCRIPT, STEP2_SCRIPT, STEP3_SCRIPT, STEP4_SCRIPT, STEP5_SCRIPT
))
if MODE_ENV in {"public", "demo", "online"}:
    PUBLIC_MODE = True
elif MODE_ENV in {"local", "research"}:
    PUBLIC_MODE = False
elif LOCAL_PIPELINE_AVAILABLE:
    with st.sidebar:
        mode_choice = st.radio(
            "运行模式",
            ["本地研究模式", "公开演示模式"],
            index=0,
            key="step6_dual_mode_choice",
        )
    PUBLIC_MODE = mode_choice == "公开演示模式"
else:
    # Cloud deployment contains only the app and aggregated public snapshot.
    PUBLIC_MODE = True

if PUBLIC_MODE:
    STEP1_OUT = PUBLIC_DATA_DIR
    STEP1_SUMMARY = PUBLIC_DATA_DIR / "candidate_edges.csv"
    STEP1_REQUIRED = [STEP1_SUMMARY]
    STEP3_OUT = PUBLIC_CACHE_DIR
    STEP4_OUT = PUBLIC_DATA_DIR / "step4_case_outputs"
    STEP5_OUT = PUBLIC_DATA_DIR / "step5_validation_outputs"
    STEP5_REPAIRED_OUT = PUBLIC_DATA_DIR / "step5_validation_outputs_repaired"
    GAMMA_OUT = PUBLIC_DATA_DIR / "gamma_sensitivity_runs"

if "step6_config_loaded" not in st.session_state:
    st.session_state.step6_config_loaded = True
    st.session_state.saved_config = read_json(CONFIG_PATH, {}) or {}
if "step6_state_loaded" not in st.session_state:
    st.session_state.step6_state_loaded = True
    st.session_state.workflow_state = load_state()

saved_cfg = st.session_state.saved_config
with st.sidebar:
    st.markdown("""
    <div class="paper-sidebar-rule"></div>
    <div class="paper-nav-title">WORKSPACE NAVIGATION</div>
    """, unsafe_allow_html=True)
    page_labels = {"证据构建": "01   需求与技术证据", "参数与优化": "02   协同方案比较", "人机评审": "03   人机协同评审", "案例分析": "04   协同改进成果", "实验验证": "05   方案稳健性"}
    page = radio_compat("导航", ["证据构建", "参数与优化", "人机评审", "案例分析", "实验验证"], index=2 if load_state().get("active_solution_path") else 0, format_func=lambda x: page_labels[x], label_visibility="collapsed", key="step6_page")
    if not PUBLIC_MODE:
        with st.expander("研究者设置", expanded=False):
            data_dir_text = st.text_input("数据目录", value=saved_cfg.get("data_dir", str(DEFAULT_DATA_DIR)), key="cfg_data_dir")
            review_text = st.text_input("统一创新单元目录", value=saved_cfg.get("review_dir", str(DEFAULT_DATA_DIR / "统一创新单元final")), key="cfg_review_dir")
            patent_text = st.text_input("专利文件", value=saved_cfg.get("patent_file", str(DEFAULT_DATA_DIR / "专利.xlsx")), key="cfg_patent_file")
            python_exe_text = st.text_input("Python解释器", value=saved_cfg.get("python_exe", sys.executable), key="cfg_python_exe")
            proxy_text = st.text_input("LLM本地代理", value=saved_cfg.get("proxy", DEFAULT_PROXY), key="cfg_proxy")
            base_url_text = st.text_input("LLM API Base URL", value=saved_cfg.get("base_url", DEFAULT_BASE_URL), key="cfg_base_url")
            model_text = st.text_input("LLM模型", value=saved_cfg.get("model", DEFAULT_MODEL), key="cfg_model")
            api_key_input = st.text_input("LLM API Key（仅本次会话使用）", value="", type="password", placeholder="留空则读取环境变量 LLM_API_KEY", key="cfg_api_key")
            if button_compat("保存路径与运行设置", key="save_cfg"):
                cfg = {
                    "data_dir": data_dir_text,
                    "review_dir": review_text,
                    "patent_file": patent_text,
                    "python_exe": python_exe_text,
                    "proxy": proxy_text,
                    "base_url": base_url_text,
                    "model": model_text,
                }
                write_json(CONFIG_PATH, cfg)
                st.session_state.saved_config = cfg
                st.success("设置已保存（API Key不会保存）。")
    else:
        data_dir_text = str(PUBLIC_DATA_DIR)
        review_text = str(PUBLIC_DATA_DIR)
        patent_text = str(PUBLIC_DATA_DIR / "candidate_edges.csv")
        python_exe_text = sys.executable
        proxy_text = DEFAULT_PROXY
        base_url_text = DEFAULT_BASE_URL
        model_text = DEFAULT_MODEL
        api_key_input = ""

if PUBLIC_MODE:
    data_dir = PUBLIC_DATA_DIR
    review_dir = PUBLIC_DATA_DIR
    patent_file = PUBLIC_DATA_DIR / "candidate_edges.csv"
else:
    data_dir = resolve_path(data_dir_text, SCRIPT_DIR)
    review_dir = resolve_path(review_text, data_dir)
    patent_file = resolve_path(patent_text, data_dir)
python_exe = python_exe_text.strip() or sys.executable
proxy = proxy_text.strip() or DEFAULT_PROXY
base_url = base_url_text.strip() or DEFAULT_BASE_URL
model = model_text.strip() or DEFAULT_MODEL
api_key = api_key_input.strip() or os.getenv("LLM_API_KEY", "").strip()

st.markdown("""
<style>
:root{--navy:#102A43;--navy2:#173F5F;--teal:#2563EB;--teal2:#2563EB;--ink:#243B53;--muted:#718397;--line:#E4EBF1;--canvas:#F4F7FA;--white:#FFFFFF}
.stApp{background:var(--canvas);color:var(--ink)}
.block-container{padding-top:1.25rem;padding-bottom:3rem;max-width:1580px}
header[data-testid="stHeader"]{background:rgba(244,247,250,.82)}
footer{visibility:hidden}
#MainMenu{visibility:hidden}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#102A43 0%,#153B56 58%,#174A5A 100%);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"]>div:first-child{background:transparent}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#D7E4ED!important}
.sidebar-brand{display:flex;align-items:center;gap:12px;padding:12px 8px 18px}
.brand-symbol{width:42px;height:42px;border-radius:13px;background:linear-gradient(145deg,#48B8AA,#167E83);display:flex;align-items:center;justify-content:center;color:white;font-size:22px;font-weight:800;box-shadow:0 8px 20px rgba(0,0,0,.2)}
.brand-name{color:#F5FAFC;font-size:13px;font-weight:800;letter-spacing:2px}
.brand-caption{color:#8FB3C4;font-size:9px;letter-spacing:2.1px;margin-top:3px}
.sidebar-divider{height:1px;background:rgba(255,255,255,.13);margin:2px 4px 18px}
.sidebar-nav-caption{font-size:9px;letter-spacing:1.8px;color:#82A6B8;margin:0 0 8px 10px;font-weight:700}
.sidebar-decision-pill{display:flex;align-items:center;gap:8px;margin:8px 2px 10px;padding:10px 11px;border-radius:9px;background:rgba(255,255,255,.09);color:#EAF3F7;font-size:11px;font-weight:750}.sidebar-decision-pill span{width:9px;height:9px;border-radius:50%;display:inline-block}
[data-testid="stSidebar"] [data-testid="stRadio"]>label{display:none}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:7px}
[data-testid="stSidebar"] div[role="radiogroup"] label{border:1px solid transparent;border-radius:11px;padding:9px 11px;background:rgba(255,255,255,.025);transition:all .18s ease}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(255,255,255,.09);border-color:rgba(255,255,255,.08)}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:linear-gradient(100deg,rgba(43,169,154,.25),rgba(43,169,154,.10));border-color:rgba(103,213,198,.28);box-shadow:inset 3px 0 0 #43B8AA}
[data-testid="stSidebar"] div[role="radiogroup"] label p{font-size:13px;font-weight:650;color:#EAF3F7!important;letter-spacing:.2px}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid rgba(255,255,255,.14);border-radius:12px;background:rgba(255,255,255,.045)}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{color:#EAF3F7}
[data-testid="stSidebar"] input{border-radius:8px}
[data-testid="stSidebar"] [data-testid="stButton"] button{background:#1C5365;color:white;border:1px solid rgba(255,255,255,.16)}
.hero-shell{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;min-height:164px;padding:28px 34px;margin:0 0 17px;border-radius:22px;background:linear-gradient(112deg,#102A43 0%,#17445E 56%,#176A70 100%);box-shadow:0 16px 36px rgba(16,42,67,.13);color:white}
.hero-shell:before{content:"";position:absolute;width:430px;height:430px;right:7%;top:-250px;border-radius:50%;border:1px solid rgba(255,255,255,.10);box-shadow:0 0 0 34px rgba(255,255,255,.025),0 0 0 70px rgba(255,255,255,.02)}
.hero-shell:after{content:"";position:absolute;width:260px;height:260px;right:18%;bottom:-210px;border-radius:50%;background:rgba(71,194,177,.14);filter:blur(5px)}
.hero-copy,.hero-side{position:relative;z-index:1}
.hero-eyebrow{font-size:9px;letter-spacing:2.3px;color:#9BC9D1;font-weight:750;margin-bottom:12px}
.hero-heading{font-size:30px;line-height:1.2;font-weight:780;letter-spacing:.2px;color:#FFFFFF}
.hero-desc{font-size:13px;color:#C4D9E3;margin-top:9px;max-width:690px;letter-spacing:.1px}
.hero-meta{display:flex;align-items:center;gap:8px;margin-top:18px;color:#D6EAEF;font-size:10px;letter-spacing:.8px}
.hero-live-dot{width:7px;height:7px;background:#69D6B8;border-radius:50%;box-shadow:0 0 0 4px rgba(105,214,184,.13)}
.hero-meta-sep{color:#76A4B0}
.hero-side{text-align:right;min-width:100px}
.hero-index{font-size:66px;line-height:1;font-weight:250;color:rgba(255,255,255,.20);letter-spacing:-4px}
.hero-side-label{font-size:9px;letter-spacing:2.4px;color:#A9CDD2;margin-top:6px}
.workflow-rail{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin:0 0 25px}
.workflow-step{display:flex;align-items:center;gap:10px;min-height:54px;padding:9px 12px;border:1px solid #E5EBF0;border-radius:13px;background:#FFFFFF;box-shadow:0 3px 10px rgba(25,52,75,.025)}
.workflow-step-index{width:27px;height:27px;flex:0 0 27px;display:flex;align-items:center;justify-content:center;border-radius:9px;background:#F0F3F6;color:#7B8B9B;font-size:10px;font-weight:800}
.workflow-step-name{font-size:12px;font-weight:700;color:#6B7E90;white-space:nowrap}
.workflow-step.done{background:#F3FAF8;border-color:#CBE8E0}
.workflow-step.done .workflow-step-index{background:#DDF3EC;color:#087C70}
.workflow-step.done .workflow-step-name{color:#176B65}
.workflow-step.current{border-color:#69BDB2;box-shadow:0 0 0 3px rgba(22,140,131,.08),0 5px 16px rgba(25,52,75,.04)}
.workflow-step.current .workflow-step-index{background:#2563EB;color:#FFFFFF}
.workflow-step.current .workflow-step-name{color:#173F5F}
.section-title{font-size:10px;font-weight:800;color:#7890A3;letter-spacing:1.7px;text-transform:uppercase;margin:5px 0 15px;padding-left:10px;border-left:3px solid #2563EB;line-height:1.5}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0 20px}
.metric-card{position:relative;overflow:hidden;background:#FFFFFF;border:1px solid #E5EBF0;border-radius:15px;padding:15px 17px 14px;box-shadow:0 4px 13px rgba(25,52,75,.035)}
.metric-card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#2563EB,#6AB9C2)}
.metric-label{font-size:11px;color:#75889A;font-weight:650;letter-spacing:.25px}
.metric-value{font-size:25px;line-height:1.25;color:#173B56;font-weight:780;margin-top:7px;letter-spacing:-.4px;overflow-wrap:anywhere}
.metric-note{font-size:10px;color:#9AA8B5;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.status-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 11px;border-radius:999px;font-size:11px;font-weight:750;letter-spacing:.1px;color:var(--pill-color);background:var(--pill-bg);border:1px solid color-mix(in srgb,var(--pill-color) 22%,white)}
.status-pill i{font-style:normal;font-size:10px}
.figure-card-title{display:flex;flex-direction:column;gap:4px;margin:8px 0 8px;padding:11px 13px;background:#FFFFFF;border:1px solid #E5EBF0;border-bottom:0;border-radius:13px 13px 0 0}
.figure-card-title span{font-size:9px;letter-spacing:1.5px;color:#2563EB;font-weight:800}
.figure-card-title b{font-size:12px;color:#29445B;font-weight:700;text-transform:capitalize}
[data-testid="stImage"] img{border-radius:0 0 13px 13px;border:1px solid #E5EBF0;border-top:0;background:#FFFFFF}
[data-testid="stPlotlyChart"]{background:#FFFFFF;border:1px solid #E5EBF0;border-radius:16px;padding:8px 8px 2px;box-shadow:0 4px 14px rgba(25,52,75,.035);margin:7px 0 15px}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #E5EBF0;padding:14px 16px;border-radius:14px;box-shadow:0 4px 13px rgba(25,52,75,.035)}
[data-testid="stDataFrame"]{border:1px solid #E3EAF0;border-radius:13px;background:#FFFFFF}
[data-testid="stExpander"]{border:1px solid #E2E9EF;border-radius:12px;background:rgba(255,255,255,.72)}
div[data-baseweb="select"]>div{border-radius:10px!important;border-color:#D8E2E9!important;background:#FFFFFF}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stTextArea"] textarea{border-radius:10px!important;border-color:#D8E2E9!important;background:#FFFFFF}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;border:1px solid #D5E0E8;font-weight:700;min-height:2.55rem;transition:all .16s ease;background:#FFFFFF;color:#29445B}
div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:#2563EB;color:#126E68;box-shadow:0 4px 12px rgba(22,140,131,.10)}
div.stButton>button[kind="primary"]{background:linear-gradient(110deg,#1D4ED8,#2563EB);border-color:#2563EB;color:white;box-shadow:0 5px 14px rgba(22,140,131,.18)}
div.stButton>button[kind="primary"]:hover{background:linear-gradient(110deg,#125B61,#117A73);color:#FFFFFF}
[data-testid="stAlert"]{border-radius:12px}
hr{border-color:#E5EBF0}
/* Figure 2-aligned human-in-the-loop review page */
.paper-review-header{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin:4px 0 14px;padding:2px 0 0}
.paper-review-breadcrumb{font-size:9px;letter-spacing:1.6px;color:#3475D8;font-weight:800;margin:0 0 10px}
.paper-review-title{font-size:29px;line-height:1.18;font-weight:800;color:#202C3A;letter-spacing:.1px}
.paper-review-subtitle{font-size:13px;color:#77869A;margin-top:7px;line-height:1.5}
.paper-review-status{display:inline-flex;align-items:center;gap:8px;white-space:nowrap;margin-top:10px;padding:9px 14px;border-radius:10px;font-size:12px;font-weight:750}
.paper-review-status span{width:9px;height:9px;border-radius:50%;background:currentColor}
.paper-review-status.pending{color:#14845E;background:#E8F6F0}
.paper-review-status.accepted{color:#1D4ED8;background:#E6F5F4}
.review-metric-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin:12px 0 15px}
.review-metric-grid .metric-card{border-radius:12px;padding:12px 15px 11px;box-shadow:none}
.review-metric-grid .metric-card:before{display:none}
.review-metric-grid .metric-value{font-size:23px;color:#253244;margin-top:5px}
.review-metric-grid .metric-label{font-size:10px}
.review-metric-grid .metric-note{font-size:9px}
.review-panel-heading{background:#FFFFFF;border:1px solid #E3E9EF;border-bottom:0;border-radius:13px 13px 0 0;padding:13px 15px 8px}
.review-panel-title{font-size:14px;font-weight:800;color:#26374A}
.review-panel-caption{font-size:10px;color:#7D8B9C;margin-top:4px;line-height:1.45}

.review-evidence-note{margin-top:7px;padding:9px 11px;border-radius:8px;background:#FFF4E6;color:#9A5A16;font-size:10px;line-height:1.5}
.review-feedback-shell{margin-top:15px;background:#FFFFFF;border:1px solid #E3E9EF;border-radius:13px;padding:14px 15px 15px}
.review-feedback-title{font-size:14px;font-weight:800;color:#26374A;margin-bottom:4px}
.review-feedback-caption{font-size:10px;color:#7D8B9C;margin-bottom:11px}
.review-subpanel{background:#F7F9FB;border:1px solid #E7EDF2;border-radius:10px;padding:11px 12px;height:100%}
.review-subpanel-title{font-size:10px;font-weight:750;color:#65768A;margin-bottom:8px}
.review-action-row{margin-top:9px}
@media(max-width:1000px){.review-metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.paper-review-header{flex-direction:column}.paper-review-status{margin-top:0}}
@media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem}.hero-shell{padding:23px 22px}.hero-heading{font-size:24px}.hero-side{min-width:60px}.hero-index{font-size:48px}.workflow-rail{grid-template-columns:repeat(2,minmax(0,1fr))}.workflow-step:last-child{grid-column:span 2}}
</style>
""", unsafe_allow_html=True)

# Figure 2 design tokens. Values mirror THEME in generate_human_interface_figure.py.
# The stylesheet is embedded below so deployment needs no extra CSS or theme file.
FIGURE2_THEME = {
    "bg": "#F4F6F8",
    "sidebar": "#182536",
    "sidebar_active": "#2A4563",
    "white": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#667085",
    "border": "#E4E7EC",
    "blue": "#2563EB",
    "blue_light": "#EAF2FF",
    "green": "#16835D",
    "green_light": "#E8F6EF",
    "amber": "#D97706",
    "amber_light": "#FFF4E5",
    "light": "#F9FAFB",
}
FIGURE2_THEME_CSS = r"""
:root{
  --paper-bg:#F4F6F8;
  --paper-sidebar:#182536;
  --paper-sidebar-active:#2A4563;
  --paper-blue:#2563EB;
  --paper-blue-light:#EAF2FF;
  --paper-green:#16835D;
  --paper-green-light:#E8F6EF;
  --paper-amber:#D97706;
  --paper-amber-light:#FFF4E5;
  --paper-text:#1F2937;
  --paper-muted:#667085;
  --paper-border:#E4E7EC;
  --paper-light:#F9FAFB;
  --paper-head:#F2F4F7;
  --paper-row:#FBFCFD;
}
.stApp{background:var(--paper-bg)!important;color:var(--paper-text)!important}
.block-container{
  max-width:none!important;
  padding-top:1.35rem!important;
  padding-bottom:2.35rem!important;
  padding-left:2rem!important;
  padding-right:2rem!important;
}
header[data-testid="stHeader"]{background:rgba(244,246,248,.92)!important}
footer{visibility:hidden!important}
#MainMenu{visibility:hidden!important}

/* Sidebar: same dark slate navigation and blue active marker as Figure 2. */
[data-testid="stSidebar"]{
  background:var(--paper-sidebar)!important;
  min-width:300px!important;
  max-width:300px!important;
  border-right:1px solid rgba(255,255,255,.08)!important;
}
[data-testid="stSidebar"]>div:first-child{
  background:transparent!important;
  padding-top:1.05rem!important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{
  color:#D9E2EC!important;
}
.paper-sidebar-brand{
  display:flex;align-items:center;gap:12px;
  padding:8px 8px 18px;
}
.paper-brand-mark{
  width:50px;height:50px;flex:0 0 50px;
  border-radius:12px;background:#2D6CDF;color:#FFFFFF;
  display:flex;align-items:center;justify-content:center;
  font-size:27px;font-weight:800;
}
.paper-brand-name{
  color:#F7FAFC;font-size:12px;font-weight:800;
  letter-spacing:.1px;line-height:1.15;white-space:nowrap;
}
.paper-brand-sub{
  color:#AAB8C8;font-size:9px;letter-spacing:.5px;margin-top:4px;
}
.paper-sidebar-rule{
  height:1px;background:rgba(255,255,255,.16);
  margin:0 4px 18px;
}
.paper-nav-title{
  color:#9AAABD;font-size:9px;letter-spacing:1.3px;
  font-weight:750;margin:0 0 10px 8px;
}
[data-testid="stSidebar"] [data-testid="stRadio"]>label{display:none!important}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:7px!important}
[data-testid="stSidebar"] div[role="radiogroup"] label{
  min-height:48px!important;
  display:flex!important;align-items:center!important;
  padding:8px 10px!important;
  border:0!important;border-left:4px solid transparent!important;
  border-radius:9px!important;background:transparent!important;
  box-shadow:none!important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background:rgba(255,255,255,.06)!important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background:var(--paper-sidebar-active)!important;
  border-left-color:var(--paper-blue)!important;
  box-shadow:none!important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label p{
  color:#D9E2EC!important;font-size:13px!important;font-weight:600!important;
  letter-spacing:.1px!important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{
  color:#FFFFFF!important;font-weight:750!important;
}
.paper-sidebar-status{
  margin-top:25px;border-top:1px solid rgba(255,255,255,.16);
  padding:18px 8px 0;color:#AAB8C8;font-size:11px;line-height:1.5;
}
.paper-status-pill{
  display:flex;align-items:center;gap:8px;
  background:#243B53;color:#F1F6FB;
  padding:10px 11px;border-radius:9px;
  font-size:12px;font-weight:700;margin:10px 0 12px;
}
.paper-status-dot{
  display:inline-block;width:8px;height:8px;flex:0 0 8px;
  border-radius:50%;background:#38B27B;
}
.paper-status-line{margin-top:7px;color:#AAB8C8}
[data-testid="stSidebar"] [data-testid="stExpander"]{
  border:1px solid rgba(255,255,255,.16)!important;
  border-radius:10px!important;background:rgba(255,255,255,.035)!important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{
  color:#D9E2EC!important;font-size:12px!important;
}
[data-testid="stSidebar"] input{border-radius:8px!important}

/* One header pattern across all five modules. */
.paper-review-header{
  display:flex;align-items:flex-start;justify-content:space-between;
  gap:18px;margin:4px 0 17px;padding:2px 0 0;
}
.paper-review-breadcrumb{
  font-size:9px;letter-spacing:1.1px;color:var(--paper-blue);
  font-weight:800;margin:0 0 10px;
}
.paper-review-title{
  font-size:29px;line-height:1.18;font-weight:800;
  color:var(--paper-text);letter-spacing:.1px;
}
.paper-review-subtitle{
  font-size:13px;color:var(--paper-muted);
  margin-top:7px;line-height:1.5;
}
.paper-review-status{
  display:inline-flex;align-items:center;gap:8px;
  white-space:nowrap;margin-top:10px;padding:9px 14px;
  border-radius:10px;font-size:12px;font-weight:750;
  color:var(--paper-green);background:var(--paper-green-light);
}
.paper-review-status span{
  width:9px;height:9px;flex:0 0 9px;border-radius:50%;
  background:currentColor;
}
.paper-review-status.waiting{
  color:#667085;background:#F2F4F7;
}

/* Metric cards and panels follow the paper figure's flat, restrained treatment. */
.metric-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px;margin:12px 0 17px;
}
.metric-card{
  position:relative;overflow:hidden;background:#FFFFFF!important;
  border:1px solid var(--paper-border)!important;border-radius:12px!important;
  padding:14px 16px!important;min-height:90px;
  box-shadow:none!important;
}
.metric-card:before{display:none!important}
.metric-label,.review-metric-label{
  font-size:11px!important;color:var(--paper-muted)!important;font-weight:600!important;
}
.metric-value,.review-metric-value{
  font-size:24px!important;line-height:1.2!important;
  color:var(--paper-text)!important;font-weight:800!important;margin-top:6px!important;
}
.metric-note,.review-metric-note{
  font-size:10px!important;color:#8A96A8!important;margin-top:4px!important;
}
.review-metric-grid{
  display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;
  gap:11px!important;margin:12px 0 15px!important;
}
.review-metric-grid .metric-card{
  border-radius:12px!important;padding:12px 15px 11px!important;
  min-height:78px!important;
}
.review-metric-grid .metric-value{font-size:23px!important;margin-top:5px!important}
.review-metric-grid .metric-label{font-size:10px!important}
.review-metric-grid .metric-note{
  display:block!important;width:100%!important;text-align:left!important;
  font-size:9px!important;white-space:normal!important;
  overflow-wrap:anywhere!important;line-height:1.35!important;
}
.review-metric{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  border-radius:12px!important;padding:12px 15px!important;
  height:78px!important;box-shadow:none!important;
}

.panel,.review-panel{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  border-radius:16px!important;padding:16px 18px!important;
  margin:0 0 14px!important;box-shadow:none!important;
}
.panel-title,.review-panel-title{
  font-size:15px!important;font-weight:800!important;
  color:#263446!important;margin-bottom:4px!important;
}
.panel-caption,.panel-sub,.review-panel-sub,.review-panel-caption{
  font-size:10px!important;color:#7D899D!important;
  margin:4px 0 10px!important;line-height:1.5!important;
}
.section-kicker,.section-title{
  font-size:9px!important;letter-spacing:1.3px!important;
  color:var(--paper-blue)!important;font-weight:800!important;
  text-transform:uppercase!important;margin:5px 0 9px!important;
  padding:0!important;border:0!important;line-height:1.5!important;
}
.small-note{font-size:11px!important;color:#758197!important;line-height:1.6!important}
.figure-card-title{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  border-bottom:0!important;border-radius:13px 13px 0 0!important;
}
.figure-card-title span{color:var(--paper-blue)!important}
.figure-card-title b{color:#263446!important}
[data-testid="stImage"] img{
  border-radius:12px!important;border:1px solid var(--paper-border)!important;
}

/* Figure 2 evidence panel and table. */
.review-panel-heading{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  border-bottom:0!important;border-radius:13px 13px 0 0!important;
  padding:13px 15px 8px!important;
}


.review-panel [data-testid="stImage"] img{
  border:0!important;border-radius:0!important;
}
.review-evidence-note,.review-insight{
  margin-top:7px!important;padding:9px 11px!important;border-radius:8px!important;
  background:var(--paper-amber-light)!important;color:#8A4B08!important;
  font-size:10px!important;line-height:1.5!important;
}
.review-feedback-shell{
  margin-top:15px!important;background:#FFFFFF!important;
  border:1px solid var(--paper-border)!important;border-radius:13px!important;
  padding:14px 15px 15px!important;
}
.review-feedback-title{font-size:14px!important;font-weight:800!important;color:#263446!important;margin-bottom:4px!important}
.review-feedback-caption{font-size:10px!important;color:#7D899D!important;margin-bottom:11px!important}
.review-subpanel{
  background:var(--paper-light)!important;border:1px solid #E7EAF0!important;
  border-radius:10px!important;padding:11px 12px!important;height:100%;
}
.review-subpanel-title,.review-form-label{
  font-size:10px!important;font-weight:750!important;color:#657286!important;
  margin:7px 0 6px!important;
}
.review-param-summary{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  border-radius:9px!important;padding:10px 11px!important;margin-top:8px!important;
}

/* Same evidence-table design in the local app and public Demo. */
.paper-table,.review-table{
  width:100%!important;table-layout:fixed!important;
  border-collapse:separate!important;border-spacing:0 5px!important;
  border:0!important;outline:0!important;box-shadow:none!important;
  font-size:10px!important;color:#4A5667!important;background:transparent!important;
}
.paper-table thead,.paper-table tbody,.paper-table tr,
.review-table thead,.review-table tbody,.review-table tr{
  border:0!important;outline:0!important;box-shadow:none!important;
}
.paper-table th,.review-table th{
  background:var(--paper-head)!important;text-align:left!important;
  padding:9px 7px!important;font-weight:700!important;color:#475467!important;
  border:0!important;white-space:normal!important;
}
.paper-table th:first-child,.review-table th:first-child{border-radius:7px 0 0 7px!important}
.paper-table th:last-child,.review-table th:last-child{border-radius:0 7px 7px 0!important}
.paper-table td,.review-table td{
  background:var(--paper-row)!important;padding:11px 7px!important;
  border:0!important;border-bottom:1px solid #EDF0F3!important;
  vertical-align:middle!important;white-space:normal!important;overflow-wrap:anywhere!important;
}
.paper-table tbody tr td:first-child,.review-table tbody tr td:first-child{border-radius:7px 0 0 7px!important}
.paper-table tbody tr td:last-child,.review-table tbody tr td:last-child{border-radius:0 7px 7px 0!important}
.paper-table .priority,.review-table .priority{color:var(--paper-blue)!important;font-weight:800!important}
.paper-table .weak-tech,.review-table .weak-tech{color:#E58A28!important;font-weight:800!important}

/* Charts, data tables and controls. */
[data-testid="stPlotlyChart"]{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  border-radius:13px!important;padding:6px!important;
  box-shadow:none!important;margin:7px 0 14px!important;
}
[data-testid="stDataFrame"]{
  border:1px solid var(--paper-border)!important;border-radius:11px!important;
  background:#FFFFFF!important;overflow:hidden!important;
}
[data-testid="stMetric"]{
  background:#FFFFFF!important;border:1px solid var(--paper-border)!important;
  padding:13px 15px!important;border-radius:12px!important;box-shadow:none!important;
}
[data-testid="stExpander"]{
  border:1px solid var(--paper-border)!important;border-radius:10px!important;
  background:#FFFFFF!important;
}
div[data-baseweb="select"]>div{
  border-radius:8px!important;border-color:#D0D5DD!important;background:#FFFFFF!important;
}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea{
  border-radius:8px!important;border-color:#D0D5DD!important;background:#FFFFFF!important;
  color:var(--paper-text)!important;
}
div.stButton>button,div.stDownloadButton>button,
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{
  border-radius:9px!important;border:1px solid #D0D5DD!important;
  font-weight:700!important;min-height:2.55rem!important;
  background:#FFFFFF!important;color:#344054!important;box-shadow:none!important;
}
div.stButton>button:hover,div.stDownloadButton>button:hover,
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover{
  border-color:var(--paper-blue)!important;color:var(--paper-blue)!important;
}
div.stButton>button[kind="primary"],
[data-testid="stButton"] button[kind="primary"]{
  background:var(--paper-blue)!important;border-color:var(--paper-blue)!important;
  color:#FFFFFF!important;
}
div.stButton>button[kind="primary"]:hover,
[data-testid="stButton"] button[kind="primary"]:hover{
  background:#1D4ED8!important;color:#FFFFFF!important;
}
[data-testid="stAlert"]{border-radius:10px!important}
hr{border-color:var(--paper-border)!important}
@media(max-width:1050px){
  .review-metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
  .paper-review-header{flex-direction:column!important}
  .paper-review-status{margin-top:0!important}
}
@media(max-width:900px){
  .block-container{padding-left:1rem!important;padding-right:1rem!important}
  [data-testid="stSidebar"]{min-width:260px!important;max-width:260px!important}
}

/* Final Figure 2 alignment fixes: keep sidebar labels legible and controls blue. */
:root{--primary-color:#2563EB}
[data-testid="stSidebar"] div[role="radiogroup"] label,
[data-testid="stSidebar"] div[role="radiogroup"] label *,
[data-testid="stSidebar"] div[role="radiogroup"] label div,
[data-testid="stSidebar"] div[role="radiogroup"] label span,
[data-testid="stSidebar"] div[role="radiogroup"] label p{
  color:#C6D1DE!important;opacity:1!important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) *{
  color:#FFFFFF!important;opacity:1!important;
}
[data-testid="stSidebar"] input[type="radio"]{accent-color:#2563EB!important}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background:#2A4563!important;border-left:4px solid #2563EB!important;
}
.paper-panel-static{
  background:#FFFFFF!important;border:1px solid #E4E7EC!important;
  border-radius:13px!important;padding:13px 15px 12px!important;
  margin:0 0 14px!important;box-shadow:none!important;
}
.review-panel-heading{
  background:#FFFFFF!important;border:1px solid #E4E7EC!important;
  border-bottom:0!important;border-radius:13px 13px 0 0!important;
  padding:13px 15px 9px!important;margin:0!important;
}
.review-panel-title{font-size:15px!important;font-weight:800!important;color:#1F2937!important}
.review-panel-caption{font-size:10px!important;color:#667085!important;margin-top:4px!important;line-height:1.5!important}
.review-panel-static-body{background:#FFFFFF!important;padding:5px 10px 11px!important}
.review-feedback-heading{
  background:#FFFFFF!important;border:1px solid #E4E7EC!important;
  border-radius:13px!important;padding:14px 15px 11px!important;
  margin:15px 0 10px!important;
}
.review-feedback-title{font-size:15px!important;font-weight:800!important;color:#1F2937!important;margin-bottom:4px!important}
.review-feedback-caption{font-size:10px!important;color:#667085!important;margin-bottom:0!important}
[data-testid="stPlotlyChart"]{
  background:#FFFFFF!important;border:1px solid #E4E7EC!important;
  border-radius:13px!important;padding:6px!important;
  box-shadow:none!important;margin:5px 0 14px!important;
}
[data-testid="stImage"] img{
  border:1px solid #E4E7EC!important;border-radius:0 0 13px 13px!important;
  border-top:0!important;background:#FFFFFF!important;
}
.review-network-empty{
  background:#FFFFFF!important;border:1px solid #E4E7EC!important;
  border-top:0!important;border-radius:0 0 13px 13px!important;
  padding:28px 16px!important;color:#667085!important;text-align:center!important;
}
.review-subpanel-title{
  font-size:10px!important;font-weight:750!important;color:#657286!important;
  margin:7px 0 6px!important;
}
.review-insight{
  margin-top:8px!important;padding:9px 11px!important;border-radius:8px!important;
  background:#FFF4E5!important;color:#8A4B08!important;
  font-size:10px!important;line-height:1.5!important;
}

/* Human-review network: one border is applied to the keyed outer container only. */
.st-key-review_network_outer{
  width:100%!important;max-width:100%!important;min-width:0!important;
  box-sizing:border-box!important;overflow:hidden!important;
  background:#FFFFFF!important;border:1px solid #E4E7EC!important;
  border-radius:13px!important;padding:12px 13px 8px!important;
}
.review-network-card-heading{
  width:100%!important;max-width:100%!important;min-width:0!important;
  box-sizing:border-box!important;margin:0 0 3px!important;
}
.st-key-review_network_outer [data-testid="stPlotlyChart"]{
  width:100%!important;max-width:100%!important;min-width:0!important;
  box-sizing:border-box!important;overflow:hidden!important;
  background:transparent!important;border:0!important;outline:0!important;
  box-shadow:none!important;border-radius:0!important;padding:0!important;
  margin:2px 0 0!important;
}
.st-key-review_network_outer [data-testid="stPlotlyChart"] *{
  box-sizing:border-box!important;border:0!important;outline:0!important;
  box-shadow:none!important;
}
.st-key-review_network_outer [data-testid="stPlotlyChart"] > div,
.st-key-review_network_outer [data-testid="stPlotlyChart"] .js-plotly-plot,
.st-key-review_network_outer [data-testid="stPlotlyChart"] .plot-container,
.st-key-review_network_outer [data-testid="stPlotlyChart"] .svg-container{
  max-width:100%!important;box-sizing:border-box!important;
}
div[data-testid="stPlotlyChart"]{
  width:100%!important;max-width:100%!important;min-width:0!important;
  box-sizing:border-box!important;overflow:hidden!important;
}
div[data-testid="stPlotlyChart"] > div,
div[data-testid="stPlotlyChart"] .js-plotly-plot,
div[data-testid="stPlotlyChart"] .plot-container,
div[data-testid="stPlotlyChart"] .svg-container{
  max-width:100%!important;box-sizing:border-box!important;
}
div[data-testid="column"]{min-width:0!important;box-sizing:border-box!important}
.review-metric-grid .metric-card{min-width:0!important;text-align:left!important}

"""
st.markdown("<style>" + FIGURE2_THEME_CSS + "</style>", unsafe_allow_html=True)

if go is None:
    st.error("交互图表需要Plotly。请在当前虚拟环境执行：python -m pip install 'plotly>=5.10,<6' --proxy http://127.0.0.1:4780")
    st.caption(PLOTLY_IMPORT_ERROR)


# Common workflow state
if PUBLIC_MODE:
    prepare_public_demo_data()
    step1_info = public_step1_info()
    cache = scan_cached_solutions(step1_info["fingerprint"], review_dir=None, data_dir=None)
    for record in cache:
        record["input_status"] = "demo_snapshot"
        record["source"] = "公开演示方案"
    workflow_state = load_state()
    workflow_state.setdefault("step1_loaded", True)
    workflow_state.setdefault("step1_confirmed", False)
    workflow_state.setdefault("step1_fingerprint", step1_info["fingerprint"])
    if not workflow_state.get("active_solution_path") or not Path(workflow_state.get("active_solution_path", "")).exists():
        default_record = find_cached_solution(12, 4, 1.0, cache) or default_solution(cache)
        if default_record:
            workflow_state = activate_solution(
                Path(default_record["path"]), review_dir, STEP1_SUMMARY,
                "公开演示基准方案", data_dir=data_dir
            )
            workflow_state["step1_loaded"] = True
            workflow_state["step1_confirmed"] = False
            workflow_state["step1_fingerprint"] = step1_info["fingerprint"]
            save_state(workflow_state)
    else:
        save_state(workflow_state)
else:
    workflow_state = load_state()
    step1_info = step1_status(data_dir, review_dir, patent_file)
    if workflow_state.get("step1_fingerprint") and workflow_state.get("step1_fingerprint") != step1_info["fingerprint"]:
        workflow_state["step1_confirmed"] = False
        workflow_state["solution_accepted"] = False
        workflow_state["active_input_binding_status"] = "stale"
        workflow_state["step4_out_dir"] = ""
        workflow_state["step5_out_dir"] = ""
        save_state(workflow_state)
    cache = scan_cached_solutions(step1_info["fingerprint"], review_dir=review_dir, data_dir=data_dir)

active_json_path, active_json, workflow_state = active_solution()
active_hash = sha256_file(active_json_path) if active_json_path and active_json_path.exists() else ""

render_app_hero(page, workflow_state, step1_info, active_json_path, active_json)

# -----------------------------------------------------------------------------
# Page 1 - Evidence construction
# -----------------------------------------------------------------------------
if page == "证据构建":
    status_badge(step1_info["status"], step1_info["label"])
    evidence_pairs = build_candidate_edges() if STEP1_SUMMARY.exists() else pd.DataFrame()
    evidence_products = (set(evidence_pairs["product_a"]) | set(evidence_pairs["product_b"])) if not evidence_pairs.empty else set()
    metric_row([
        ("候选协同关系", len(evidence_pairs) if not evidence_pairs.empty else "—", "来自用户反馈"),
        ("涉及产品", len(evidence_products) if evidence_products else "—", "跨产品协同网络"),
        ("可追溯证据", "需求 + 技术", "评论语义与专利知识"),
    ])
    if PUBLIC_MODE:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("**需求—技术证据**")
            st.write("候选产品关系由消费者需求信息与专利技术证据共同支持。")
            st.success("案例数据已加载，可查看候选关系及其依据。")
        with c2:
            st.markdown("**证据确认**")
            if workflow_state.get("step1_confirmed"):
                st.success("需求与技术证据已确认。")
            else:
                st.write("核对候选关系后，确认进入协同方案比较。")
                if button_compat("确认需求与技术证据，进入优化", key="public_confirm_step1", type="primary"):
                    workflow_state["step1_loaded"] = True
                    workflow_state["step1_confirmed"] = True
                    workflow_state["step1_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
                    workflow_state["step1_fingerprint"] = step1_info["fingerprint"]
                    save_state(workflow_state)
                    st.success("证据已确认。")
                    rerun_app()
    else:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("**需求—技术证据状态**")
            with st.expander("查看输入和输出文件", expanded=False):
                st.write(f"评论数据：`{review_dir}`")
                st.write(f"专利文件：`{patent_file}`")
                st.write(f"Step1输出：`{STEP1_OUT}`")
            if step1_info["missing"]:
                st.warning("缺少输出：" + "、".join(Path(x).name for x in step1_info["missing"]))
            elif step1_info["status"] == "legacy":
                st.info("检测到本地历史Step1结果。当前没有Step6生成的输入指纹，建议核对数据版本后确认复用。")
            elif step1_info["status"] == "stale":
                st.warning("输入数据或脚本已变化，现有Step1结果可能过期。")
            else:
                st.success("Step1结果完整，可直接加载。")
        with c2:
            st.markdown("**运行操作**")
            if button_compat("加载并使用已有Step1结果", key="use_step1", disabled=bool(step1_info["missing"]), type="primary"):
                workflow_state = load_state()
                workflow_state["step1_loaded"] = True
                workflow_state["step1_fingerprint"] = step1_info["fingerprint"]
                workflow_state["step1_confirmed"] = False
                save_state(workflow_state)
                st.success("已加载。请核对下方候选关系后确认。")
            if button_compat("运行 / 重新运行 Step1", key="run_step1"):
                with st.spinner("正在执行Step1…"):
                    ok, msg = run_step1(data_dir, review_dir, patent_file, proxy, api_key, base_url, model, python_exe)
                if ok:
                    st.success(msg)
                    workflow_state = load_state()
                    workflow_state.update({"step1_loaded": True, "step1_confirmed": False, "step1_fingerprint": step1_status(data_dir, review_dir, patent_file)["fingerprint"]})
                    if workflow_state.get("active_solution_path"):
                        workflow_state["active_input_binding_status"] = "stale"
                        workflow_state["solution_accepted"] = False
                    save_state(workflow_state)
                else:
                    st.error(msg)
            if workflow_state.get("step1_loaded") and not workflow_state.get("step1_confirmed") and STEP1_SUMMARY.exists():
                if button_compat("确认需求与技术证据，进入优化", key="confirm_step1", type="primary"):
                    workflow_state["step1_confirmed"] = True
                    workflow_state["step1_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
                    workflow_state["step1_fingerprint"] = step1_info["fingerprint"]
                    write_json(MANIFEST_DIR / "step1_manifest.json", {"fingerprint": step1_info["fingerprint"], "created_at": datetime.now().isoformat(timespec="seconds"), "verification": "user_confirmed_existing_or_current_results"})
                    save_state(workflow_state)
                    st.success("Step1证据已确认。可以进入参数与优化页面。")
    if STEP1_SUMMARY.exists():
        candidate = build_candidate_edges()
        st.markdown("**候选产品关系网络**")
        if not candidate.empty and go is not None:
            fig = network_figure(candidate, pd.DataFrame(), "候选网络")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
        if PUBLIC_MODE:
            show_cols = [c for c in ["product_a", "product_b", "combined_weight", "freq", "demand_norm", "tech_norm", "scenario_primary", "issue_primary"] if c in candidate.columns]
        else:
            show_cols = [c for c in ["product_a", "product_b", "pair_freq", "severity_sum", "tech_strength_proxy_score_sum", "need_tech_alignment_score_norm", "scenario_primary", "issue_primary"] if c in candidate.columns]
        dataframe_compat(candidate[show_cols].head(60), use_container_width=True, height=270)

# -----------------------------------------------------------------------------
# Page 2 - Parameter selection and optimization
# -----------------------------------------------------------------------------
elif page == "参数与优化":
    if not workflow_state.get("step1_confirmed"):
        st.warning("请先在「证据构建」页面确认Step1结果。")
    else:
        if not cache:
            if PUBLIC_MODE:
                st.warning("当前演示数据中没有可用的参数方案。")
            else:
                st.info("暂未发现可复用的Step2/3结果。选择K、B和γ后，可直接调用Step3固定参数模式，由Step3调用Step2精确求解。")
        elif go is not None:
            search_fig = parameter_scatter(cache)
            if search_fig is not None:
                st.plotly_chart(search_fig, use_container_width=True)
        combos = sorted(set((x["k"], x["b"]) for x in cache))
        if not combos:
            combos = [(12, 4)]
        latest_default = default_solution(cache)
        default_pair = (latest_default["k"], latest_default["b"]) if latest_default else combos[0]
        default_pair = default_pair if default_pair in combos else combos[0]
        custom = False if PUBLIC_MODE else st.checkbox("自定义K/B（默认只展示已运行组合）", value=False, key="custom_kb")
        if not custom:
            labels = [f"K = {k}   ·   B = {b}" for k, b in combos]
            default_label = f"K = {default_pair[0]}   ·   B = {default_pair[1]}"
            selected_label = st.selectbox("可选K/B组合" if PUBLIC_MODE else "已运行的K/B组合", labels, index=labels.index(default_label), key="cached_kb_choice")
            k, b = combos[labels.index(selected_label)]
        else:
            observed_ks = [x[0] for x in combos] or [12]
            observed_bs = [x[1] for x in combos] or [4]
            outside = st.checkbox("允许超出已运行K/B范围（高级设置）", value=False, key="kb_outside")
            k_min, k_max = (1, 45) if outside else (min(observed_ks), max(observed_ks))
            b_min, b_max = (1, 10) if outside else (min(observed_bs), max(observed_bs))
            c1, c2 = st.columns(2)
            with c1:
                k = st.number_input("K · 产品对数量", min_value=int(k_min), max_value=int(k_max), value=int(max(k_min, min(k_max, default_pair[0]))), step=1, key="custom_k_value")
            with c2:
                b = st.number_input("B · 单产品最大参与次数", min_value=int(b_min), max_value=int(b_max), value=int(max(b_min, min(b_max, default_pair[1]))), step=1, key="custom_b_value")
        gammas_for_pair = sorted(set(round(x["gamma"], 8) for x in cache if x["k"] == int(k) and x["b"] == int(b)))
        global_gammas = sorted(set(round(x["gamma"], 8) for x in cache)) or [1.0]
        if not gammas_for_pair:
            gammas_for_pair = [1.0]
        gamma_custom = False if PUBLIC_MODE else st.checkbox("自定义γ（当前K/B没有对应历史结果时可用）", value=False, key="custom_gamma")
        if gamma_custom:
            allow_gamma_outside = st.checkbox("允许超出已运行γ范围", value=False, key="gamma_outside")
            gmin = min(global_gammas) if not allow_gamma_outside else 0.0
            gmax = max(global_gammas) if not allow_gamma_outside else 5.0
            gamma = st.number_input("γ · 技术增益系数", min_value=float(gmin), max_value=float(gmax), value=float(gammas_for_pair[0]), step=0.1, format="%.2f", key="custom_gamma_value")
        else:
            gamma_labels = [f"{g:.2f}" for g in gammas_for_pair]
            default_gamma = safe_float(latest_default.get("gamma"), 1.0) if latest_default and (latest_default["k"], latest_default["b"]) == (int(k), int(b)) else (1.0 if 1.0 in gammas_for_pair else gammas_for_pair[0])
            if default_gamma not in gammas_for_pair:
                default_gamma = gammas_for_pair[0]
            gamma_label = st.selectbox("γ · 技术增益系数", gamma_labels, index=gamma_labels.index(f"{default_gamma:.2f}"), key="cached_gamma_choice")
            gamma = float(gamma_label)
        candidate_for_feasibility = build_candidate_edges()
        node_n = len(set(candidate_for_feasibility.get("product_a", pd.Series(dtype=str)).astype(str)) | set(candidate_for_feasibility.get("product_b", pd.Series(dtype=str)).astype(str)))
        edge_n = len(candidate_for_feasibility)
        max_feasible = min(edge_n, (node_n * int(b)) // 2)
        if int(k) > max_feasible:
            st.error(f"当前K/B组合不可行：当前候选网络包含{node_n}个产品节点、{edge_n}条候选关系，B={b}时K理论上限为{max_feasible}。")
        cached = find_cached_solution(int(k), int(b), float(gamma), cache)
        if cached:
            if cached.get("input_status") == "demo_snapshot":
                status_badge("verified", "当前参数方案可用")
            elif cached.get("input_status") == "verified":
                status_badge("verified", "已找到精确匹配结果 · 输入版本一致")
            elif cached.get("input_status") == "verified_legacy":
                status_badge("verified", "历史Step3输入指纹已核验 · 可复用")
                st.info("该结果虽未保存Step6界面绑定记录，但Step3历史输入指纹、当前输入文件及Step2结果哈希均已核验一致。加载后仍需在人机评审中决定是否接受方案。")
            elif cached.get("input_status") == "stale":
                status_badge("stale", "历史结果与当前输入不一致")
                st.warning("该历史Step2结果的输入指纹与当前数据不一致，不建议直接复用。")
            else:
                status_badge("legacy", "已找到历史结果 · 输入版本待核验")
                st.warning("该历史Step2结果缺少可与当前输入对照的完整指纹，加载后需要人工核对数据版本。")
            metric_row([("目标函数值", f"{cached['weight']:.4f}", "当前参数方案" if PUBLIC_MODE else "历史精确求解"), ("入选产品对", cached["selected_count"], f"K={k}"), ("求解状态", "最优" if cached["optimal"] else "需核验", "精确优化结果" if PUBLIC_MODE else cached["source"])])
            if button_compat("加载该参数方案" if PUBLIC_MODE else "加载已有优化结果", key="load_cached_solution", type="primary"):
                state = activate_solution(Path(cached["path"]), review_dir, STEP1_SUMMARY, cached["source"], data_dir=data_dir)
                st.session_state.workflow_state = state
                st.success("已加载历史结果。请进入「人机评审」确认方案。")
        else:
            status_badge("missing", "没有找到该参数组合的匹配结果")
            if PUBLIC_MODE:
                st.info("该参数组合暂无可展示方案，请选择列表中的其他组合。")
            else:
                st.info("系统不会自动扩大默认参数范围。确认运行后，将使用当前K/B/γ调用固定参数Step3，由Step3调用原Step2精确求解。")
                if button_compat("运行 Step2/3 精确优化", key="run_fixed_step3", type="primary", disabled=int(k) > max_feasible):
                    with st.spinner(f"正在求解 K={k}, B={b}, γ={gamma:.2f} …"):
                        ok, json_path, summary_path, msg = run_fixed_step3(int(k), int(b), float(gamma), data_dir, review_dir, STEP1_SUMMARY, proxy, api_key, base_url, model, python_exe)
                    if ok and json_path:
                        current_step1_fp = step1_status(data_dir, review_dir, patent_file)["fingerprint"]
                        write_json(json_path.parent / "step6_binding.json", {"step1_fingerprint": current_step1_fp, "step1_result_sha256": step1_result_sha256(), "k": int(k), "b": int(b), "gamma": float(gamma), "step2_script_sha256": sha256_file(STEP2_SCRIPT), "step3_script_sha256": sha256_file(STEP3_SCRIPT), "created_at": datetime.now().isoformat(timespec="seconds")})
                        state = activate_solution(json_path, review_dir, STEP1_SUMMARY, "本次界面运行", summary_path, data_dir=data_dir)
                        st.session_state.workflow_state = state
                        st.success(msg)
                    else:
                        st.error(msg)
        # Parameter result map: show current active solution if it matches selected parameters.
        active_json_path, active_json, workflow_state = active_solution()
        if active_json_path and active_json:
            meta = active_json.get("meta", {})
            if safe_int(meta.get("k_select")) == int(k) and safe_int(meta.get("max_edges_per_node")) == int(b) and abs(safe_float(meta.get("gamma"), 1.0) - float(gamma)) < 1e-8:
                candidate, selected = get_current_candidate_and_selected(active_json_path)
                st.markdown("**产品协同网络**")
                view = selectbox_compat("网络显示", ["筛选对照", "优化网络", "候选网络"], key="network_view_opt", label_visibility="collapsed")
                fig = network_figure(candidate, selected, view)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
                dataframe_compat(selected, use_container_width=True, height=280)

    divider_compat()
    if PUBLIC_MODE:
        if workflow_state.get("step1_confirmed"):
            public_gamma_fig = public_gamma_figure(int(k), int(b))
            if public_gamma_fig is not None:
                with st.expander("查看技术增益系数敏感性", expanded=False):
                    st.plotly_chart(public_gamma_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        display_gamma_sensitivity_panel(key_prefix="params", k=int(k) if workflow_state.get("step1_confirmed") else None,
                                        b=int(b) if workflow_state.get("step1_confirmed") else None)

# -----------------------------------------------------------------------------
# Page 3 - Human review and feedback (Figure 2-aligned live interface)
# -----------------------------------------------------------------------------
elif page == "人机评审":
    json_path, data, workflow_state = active_solution()
    if json_path is None or data is None:
        st.info("请先在「协同方案比较」页面加载或运行一个优化方案。")
    else:
        meta = data.get("meta", {})
        selected = pd.DataFrame(data.get("selected_pairs", []) or [])
        input_binding_status = workflow_state.get("active_input_binding_status", "legacy_unverified")
        solver = meta.get("solver", {}) or {}
        selected_nodes = set(selected.get("product_a", pd.Series(dtype=str)).astype(str)) | set(selected.get("product_b", pd.Series(dtype=str)).astype(str))
        accepted_now = bool(workflow_state.get("solution_accepted") and workflow_state.get("accepted_solution_hash") == sha256_file(json_path))

        # Keep important provenance warnings visible, but subordinate to the decision content.
        if input_binding_status == "stale":
            st.error("当前优化结果对应的输入数据与现有证据不一致，不能直接确认。请重新运行优化。")
        elif input_binding_status == "legacy_unverified":
            st.warning("该历史优化结果尚未完成输入版本核验。请核对评论数据、专利汇总及技术分类版本后再决定是否接受。")
        elif input_binding_status == "verified_legacy":
            st.info("历史优化结果的输入指纹已核验一致；请继续结合产品关系与需求—技术证据进行人工评审。")

        metric_row([
            ("优先协同改进关系", len(selected), f"方案规模 K={safe_int(meta.get('k_select'))}"),
            ("协同优先级总值", f"{safe_float(solver.get('best_weight')):.4f}", "当前优化方案"),
            ("涉及产品节点", len(selected_nodes), "跨产品协同覆盖"),
            ("单产品最大参与", safe_int(meta.get("max_edges_per_node")), f"度约束 B={safe_int(meta.get('max_edges_per_node'))}"),
        ], class_name="review-metric-grid")

        candidate, _ = get_current_candidate_and_selected(json_path)
        left_panel, right_panel = st.columns([1.02, 1.0])

        with left_panel:
            # The Streamlit container supplies the single outer frame. The
            # Plotly canvas is borderless and constrained to this container.
            with st.container(key="review_network_outer"):
                st.markdown(
                    '<div class="review-network-card-heading">'
                    '<div class="review-panel-title">当前优先协同改进网络</div>'
                    '<div class="review-panel-caption">橙色节点表示当前方案中的核心参与产品；连线粗细反映综合优先权重。</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                fig = network_figure(candidate, selected, "筛选对照", paper_style=True, height=365)
                if fig is not None:
                    st.plotly_chart(
                        fig, use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": True, "responsive": True},
                    )
                else:
                    st.markdown(
                        '<div class="review-network-empty">当前没有可展示的产品协同网络。</div>',
                        unsafe_allow_html=True,
                    )

        with right_panel:
            view = selected.copy()
            table_html = (
                "<div class='review-panel-static-body'><div class='small-note'>当前方案没有入选产品对。</div></div>"
            )
            evidence_html = ""
            if not view.empty:
                product_a = view["product_a"].astype(str) if "product_a" in view.columns else pd.Series([""] * len(view), index=view.index)
                product_b = view["product_b"].astype(str) if "product_b" in view.columns else pd.Series([""] * len(view), index=view.index)
                view["产品协同关系"] = product_a.map(paper_product_label) + "—" + product_b.map(paper_product_label)
                display_map = {
                    "combined_weight": "综合优先级", "demand_norm": "需求强度", "tech_norm": "技术支撑",
                    "scenario_primary": "主要改进类别",
                }
                cols = ["产品协同关系"] + [c for c in display_map if c in view.columns]
                view = view[cols].rename(columns=display_map)
                for col in ("综合优先级", "需求强度", "技术支撑"):
                    if col in view.columns:
                        view[col] = pd.to_numeric(view[col], errors="coerce").round(3)
                if "综合优先级" in view.columns:
                    view = view.sort_values("综合优先级", ascending=False)

                table_rows = []
                for _, row in view.head(5).iterrows():
                    pair = str(row.get("产品协同关系", ""))
                    priority = safe_float(row.get("综合优先级"))
                    demand = safe_float(row.get("需求强度"))
                    tech = safe_float(row.get("技术支撑"))
                    category = str(row.get("主要改进类别", ""))
                    tech_class = "weak-tech" if tech < 0.25 else ""
                    table_rows.append(
                        "<tr>"
                        f"<td>{html_lib.escape(pair)}</td>"
                        f"<td class='priority'>{priority:.4f}</td>"
                        f"<td>{demand:.3f}</td>"
                        f"<td class='{tech_class}'>{tech:.3f}</td>"
                        f"<td>{html_lib.escape(category)}</td>"
                        "</tr>"
                    )
                table_html = (
                    "<div class='review-panel-static-body'><table class='paper-table'><colgroup>"
                    "<col style='width:29%'><col style='width:16%'><col style='width:14%'>"
                    "<col style='width:14%'><col style='width:27%'></colgroup>"
                    "<thead><tr><th>产品协同关系</th><th>综合优先级</th>"
                    "<th>需求强度</th><th>技术支撑</th><th>主要改进类别</th>"
                    "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table></div>"
                )
                if {"需求强度", "技术支撑"}.issubset(view.columns):
                    diagnostic = view.copy()
                    diagnostic["需求强度"] = pd.to_numeric(diagnostic["需求强度"], errors="coerce")
                    diagnostic["技术支撑"] = pd.to_numeric(diagnostic["技术支撑"], errors="coerce")
                    gap = diagnostic[diagnostic["需求强度"] > diagnostic["技术支撑"]]
                    if not gap.empty:
                        row = gap.sort_values("需求强度", ascending=False).iloc[0]
                        evidence_html = (
                            '<div class="review-insight"><b>需求与技术支撑的差异值得关注</b><br>'
                            + html_lib.escape(str(row["产品协同关系"]))
                            + f'：需求强度 {row["需求强度"]:.3f}，技术支撑 {row["技术支撑"]:.3f}。'
                            + '</div>'
                        )

            right_panel_html = (
                '<div class="paper-panel-static">'
                '<div class="review-panel-title">改进优先级及其需求—技术依据</div>'
                '<div class="review-panel-caption">按综合优先级排序，展示供管理者核验的主要依据。</div>'
                + table_html + evidence_html + '</div>'
            )
            st.markdown(right_panel_html, unsafe_allow_html=True)

        # Feedback and parameter adjustment retain the original workflow logic.
        st.markdown(
            '<div class="review-feedback-heading"><div class="review-feedback-title">管理者反馈与参数调整</div>'
            '<div class="review-feedback-caption">根据产品覆盖、实施负担和技术支撑审查当前组合；必要时调整参数并比较新方案。</div></div>',
            unsafe_allow_html=True,
        )
        feedback_options = ["方案过于集中", "产品覆盖不足", "产品对数量偏多", "产品对数量偏少", "技术支撑不足", "技术因素过强", "业务上暂不适用", "证据不足", "其他"]
        combos = sorted(set((int(x["k"]), int(x["b"])) for x in cache))
        current_pair = (safe_int(meta.get("k_select")), safe_int(meta.get("max_edges_per_node")))
        if not combos:
            combos = [current_pair]
        if current_pair not in combos:
            combos.append(current_pair)
            combos = sorted(set(combos))
        current_gamma = safe_float(meta.get("gamma"), 1.0)

        feedback_col, k_col, b_col, gamma_col = st.columns([1.15, 0.72, 0.72, 0.72])
        with feedback_col:
            st.markdown('<div class="review-subpanel-title">人工反馈</div>', unsafe_allow_html=True)
            reason = st.selectbox("反馈类型", feedback_options, key="review_reason")
            note = st.text_area("补充意见", key="review_note", height=92,
                                placeholder="结合产品覆盖、实施约束或技术依据记录管理判断…")
            confirm_legacy_binding = True
            if input_binding_status == "legacy_unverified":
                confirm_legacy_binding = st.checkbox(
                    "我已核对该历史优化结果与当前Step1证据相对应",
                    value=False, key="confirm_legacy_solution_binding"
                )

        k_values = sorted(set(k for k, _ in combos))
        default_k = current_pair[0] if current_pair[0] in k_values else k_values[0]
        if st.session_state.get("review_k_v263") not in k_values:
            st.session_state["review_k_v263"] = default_k
        with k_col:
            st.markdown('<div class="review-subpanel-title">K · 产品对数量</div>', unsafe_allow_html=True)
            new_k = int(st.selectbox("K · 产品对数量", k_values, key="review_k_v263"))

        b_values = sorted(b for k, b in combos if k == new_k)
        default_b = current_pair[1] if current_pair[0] == new_k and current_pair[1] in b_values else b_values[0]
        if st.session_state.get("review_b_v263") not in b_values:
            st.session_state["review_b_v263"] = default_b
        with b_col:
            st.markdown('<div class="review-subpanel-title">B · 单产品最大参与次数</div>', unsafe_allow_html=True)
            new_b = int(st.selectbox("B · 单产品最大参与次数", b_values, key="review_b_v263"))

        gamma_values = sorted(set(safe_float(x["gamma"]) for x in cache
                                  if int(x["k"]) == new_k and int(x["b"]) == new_b))
        if not gamma_values:
            gamma_values = [current_gamma]
        if current_gamma not in gamma_values:
            gamma_values.append(current_gamma)
            gamma_values = sorted(set(gamma_values))
        gamma_labels = [f"{g:.2f}" for g in gamma_values]
        gamma_default = current_gamma if current_gamma in gamma_values else gamma_values[0]
        if st.session_state.get("review_gamma_v263") not in gamma_labels:
            st.session_state["review_gamma_v263"] = f"{gamma_default:.2f}"
        with gamma_col:
            st.markdown('<div class="review-subpanel-title">γ · 技术增益系数</div>', unsafe_allow_html=True)
            new_gamma_label = st.selectbox("γ · 技术增益系数", gamma_labels, key="review_gamma_v263")
            new_gamma = float(new_gamma_label)
            st.markdown(
                '<div class="review-param-summary"><div class="review-subpanel-title">当前方案</div>'
                f'<div style="font-size:18px;font-weight:800;color:#1F2937">K={current_pair[0]} · B={current_pair[1]}</div>'
                f'<div style="font-size:11px;color:#667085;margin-top:4px">γ={current_gamma:.2f}</div></div>',
                unsafe_allow_html=True,
            )

        action1, action2 = st.columns([1, 1])
        with action1:
            if button_compat("接受当前方案", key="accept_solution", type="primary", disabled=(input_binding_status == "stale" or not confirm_legacy_binding or (PUBLIC_MODE and not workflow_state.get("step1_confirmed")))):
                workflow_state["solution_accepted"] = True
                workflow_state["accepted_at"] = datetime.now().isoformat(timespec="seconds")
                workflow_state["accepted_solution_hash"] = sha256_file(json_path)
                if input_binding_status in {"verified", "verified_legacy"}:
                    write_json(json_path.parent / "step6_binding.json", {
                        **solution_binding(json_path),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "verified_from": "legacy_step3_manifest" if input_binding_status == "verified_legacy" else "existing_step6_binding",
                    })
                    workflow_state["active_input_binding_status"] = "verified"
                save_state(workflow_state)
                append_decision_log({"timestamp": datetime.now().isoformat(timespec="seconds"), "action": "accept", "reason": "接受当前方案", "note": note, "old_k": current_pair[0], "old_b": current_pair[1], "old_gamma": current_gamma, "new_k": current_pair[0], "new_b": current_pair[1], "new_gamma": current_gamma, "solution": str(json_path)})
                st.success("方案已确认。现在可以进入协同改进成果页面。")
                rerun_app()
        with action2:
            if button_compat("按反馈切换方案" if PUBLIC_MODE else "按反馈重新优化", key="feedback_reoptimize", disabled=(input_binding_status == "stale" or not workflow_state.get("step1_confirmed"))):
                if (new_k, new_b, new_gamma) == (current_pair[0], current_pair[1], current_gamma):
                    st.warning("参数没有变化。若接受当前方案，请使用「接受当前方案」。")
                else:
                    cached = find_cached_solution(new_k, new_b, new_gamma, cache)
                    if cached:
                        new_state = activate_solution(
                            Path(cached["path"]), review_dir, STEP1_SUMMARY,
                            "反馈后切换到已有方案" if PUBLIC_MODE else "反馈后复用历史结果",
                            data_dir=data_dir
                        )
                        st.session_state.workflow_state = new_state
                        result_path = cached["path"]
                        st.success("已切换到对应参数方案。" if PUBLIC_MODE else "找到匹配历史结果，已直接加载，没有重复计算。")
                    elif PUBLIC_MODE:
                        result_path = ""
                        st.warning("该参数组合没有可用方案，请选择其他已有组合。")
                    else:
                        with st.spinner("正在按人工参数重新优化…"):
                            ok, new_json, new_summary, msg = run_fixed_step3(new_k, new_b, new_gamma, data_dir, review_dir, STEP1_SUMMARY, proxy, api_key, base_url, model, python_exe)
                        if ok and new_json:
                            current_step1_fp = step1_status(data_dir, review_dir, patent_file)["fingerprint"]
                            write_json(new_json.parent / "step6_binding.json", {"step1_fingerprint": current_step1_fp, "step1_result_sha256": step1_result_sha256(), "k": int(new_k), "b": int(new_b), "gamma": float(new_gamma), "step2_script_sha256": sha256_file(STEP2_SCRIPT), "step3_script_sha256": sha256_file(STEP3_SCRIPT), "created_at": datetime.now().isoformat(timespec="seconds")})
                            new_state = activate_solution(new_json, review_dir, STEP1_SUMMARY, "人工反馈触发的新计算", new_summary, data_dir=data_dir)
                            st.session_state.workflow_state = new_state
                            result_path = str(new_json)
                            st.success(msg)
                        else:
                            result_path = ""
                            st.error(msg)
                    if result_path:
                        append_decision_log({"timestamp": datetime.now().isoformat(timespec="seconds"), "action": "switch_cached_solution" if PUBLIC_MODE else "reoptimize", "reason": reason, "note": note, "old_k": current_pair[0], "old_b": current_pair[1], "old_gamma": current_gamma, "new_k": new_k, "new_b": new_b, "new_gamma": new_gamma, "solution": result_path})
                        st.info("新方案尚未确认。请重新检查产品网络和入选产品对。")
                        rerun_app()

        if accepted_now:
            status_badge("verified", "当前方案已由人工确认")
        else:
            status_badge("legacy", "当前方案等待人工确认")

        if PUBLIC_MODE:
            public_history = list(st.session_state.get("_step6_public_decision_log", []) or [])
            if public_history:
                with st.expander("查看本次评审记录", expanded=False):
                    history = pd.DataFrame(public_history)
                    dataframe_compat(history.tail(20).iloc[::-1], use_container_width=True, height=240)
                    st.download_button(
                        "导出评审记录",
                        data=history.to_csv(index=False).encode("utf-8-sig"),
                        file_name="decision_review_record.csv",
                        mime="text/csv",
                        key="download_public_decision_log",
                    )
        else:
            log_file = WORKSPACE_DIR / "decision_log.csv"
            if log_file.exists():
                with st.expander("查看决策历史", expanded=False):
                    try:
                        history = read_csv_any(log_file)
                        dataframe_compat(history.tail(20).iloc[::-1], use_container_width=True, height=240)
                        st.download_button("导出反馈记录", data=log_file.read_bytes(), file_name="decision_log.csv", mime="text/csv", key="download_decision_log")
                    except Exception as exc:
                        st.warning(f"历史记录读取失败：{exc}")

        gamma_fig = public_gamma_figure(current_pair[0], current_pair[1]) if PUBLIC_MODE else gamma_figure(cache, current_pair[0], current_pair[1])
        if gamma_fig is not None:
            with st.expander("查看当前 K/B 下的 γ 敏感性结果", expanded=False):
                st.plotly_chart(gamma_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 4 - Case analysis / Step4
# -----------------------------------------------------------------------------
elif page == "案例分析":
    json_path, data, workflow_state = active_solution()
    if json_path is None or data is None:
        st.warning("尚未加载优化方案。")
    elif PUBLIC_MODE:
        render_public_case_analysis(data)
    else:
        json_path, data, workflow_state = active_solution()
        if json_path is None or data is None:
            st.warning("尚未加载优化方案。")
        elif not workflow_state.get("solution_accepted") or workflow_state.get("accepted_solution_hash") != sha256_file(json_path):
            st.info("当前方案尚未通过本次人机评审。以下为可用历史案例成果的只读预览；正式接受与重新计算仍需在‘人机评审’确认。")
            legacy_tables_dir = STEP4_OUT / "tables"
            legacy_display_table = legacy_tables_dir / "table_final_selected_pairs_display.csv"
            legacy_raw_table = legacy_tables_dir / "table_final_selected_pairs.csv"
            legacy_table = legacy_display_table if legacy_display_table.exists() else legacy_raw_table
            if legacy_table.exists():
                st.markdown("**已有协同改进关系与证据**")
                dataframe_compat(decision_explanation_table(read_csv_any(legacy_table)), use_container_width=True, height=380)
                display_figures(STEP4_OUT / "figures", max_items=8)
            else:
                st.warning("未找到历史案例成果，请先核查案例分析目录。")
        else:
            s4 = get_step4_status(json_path)
            status_badge(s4["status"], s4["label"])
            if s4.get("out_dir"):
                out_dir = Path(s4["out_dir"])
            else:
                out_dir = STEP4_OUT
            if s4["status"] == "verified":
                st.success("已找到与当前最终方案绑定的Step4结果，可直接展示。")
            elif s4["status"] == "legacy":
                st.warning("检测到历史Step4结果，参数一致但尚无完整指纹。请核对后选择复用或重新运行。")
                if button_compat("复用现有Step4结果", key="reuse_legacy_step4"):
                    out_dir = Path(s4.get("out_dir", str(STEP4_OUT)))
                    write_json(out_dir / "step6_binding.json", solution_binding(json_path))
                    workflow_state["step4_out_dir"] = str(out_dir.resolve())
                    save_state(workflow_state)
                    st.success("已按人工确认记录复用。")
            elif s4["status"] == "stale":
                st.warning("现有Step4结果无法确认对应当前方案。重新运行时会保留旧目录。")
            if s4["status"] != "verified" and button_compat("运行 / 更新 Step4 案例分析", key="run_step4", type="primary"):
                with st.spinner("正在运行Step4并生成论文图表…"):
                    ok, out_dir, msg = run_step4(json_path, Path(workflow_state["active_summary_path"]), data_dir, review_dir, STEP1_SUMMARY, python_exe, proxy, api_key, base_url, model)
                if ok:
                    st.success(msg)
                    workflow_state["step4_done_hash"] = sha256_file(json_path)
                    workflow_state["step4_out_dir"] = str(out_dir.resolve())
                    save_state(workflow_state)
                else:
                    st.error(msg)
            if out_dir.exists():
                st.markdown("**案例分析图表**")
                display_figures(out_dir / "figures", max_items=12)
                tables_dir = out_dir / "tables"
                display_table = tables_dir / "table_final_selected_pairs_display.csv"
                raw_table = tables_dir / "table_final_selected_pairs.csv"
                selected_table = display_table if display_table.exists() else raw_table
                if selected_table.exists():
                    st.markdown("**最终入选产品对**")
                    dataframe_compat(decision_explanation_table(read_csv_any(selected_table)), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 5 - Step5 validation
# -----------------------------------------------------------------------------
elif page == "实验验证":
    if PUBLIC_MODE:
        render_public_robustness()
    else:
        display_gamma_sensitivity_panel(key_prefix="validation", compact=True)
        divider_compat()
        json_path, data, workflow_state = active_solution()
        if json_path is None or data is None:
            st.warning("尚未加载优化方案。")
        elif not workflow_state.get("solution_accepted") or workflow_state.get("accepted_solution_hash") != sha256_file(json_path):
            st.info("当前方案尚未在本次交互中接受。已完成的稳健性实验可只读浏览；参数与数据绑定需人工核对后才能作为当前新方案的验证。")
            s5_preview = get_step5_status(json_path)
            preview_dir = Path(s5_preview["out_dir"]) if s5_preview.get("all_complete") else None
            if preview_dir is not None:
                st.markdown("**历史完整稳健性实验**")
                st.caption(s5_preview["label"])
                table = preview_dir / "tables" / "llm_analysis_summary_by_scenario.csv"
                if table.exists():
                    dataframe_compat(read_csv_any(table), use_container_width=True, height=300)
                display_figures(preview_dir / "llm_perturb_tau_0p3" / "figures", max_items=7)
            else:
                st.warning("没有发现完整、可预览的稳健性实验。")
        else:
            s4 = get_step4_status(json_path)
            step4_ready = (s4["status"] == "verified") or (workflow_state.get("step4_done_hash") == sha256_file(json_path)) or (s4["status"] == "legacy" and workflow_state.get("step4_out_dir") == s4.get("out_dir"))
            if not step4_ready:
                st.warning("请先完成或确认复用当前方案对应的Step4案例分析，再运行Step5。")
            s5 = get_step5_status(json_path)
            status_badge(s5["status"], s5["label"])
            if s5["status"] == "verified":
                st.success("已找到与当前最终方案绑定的Step5结果。")
            elif s5["status"] == "legacy":
                st.warning("检测到历史Step5结果，但尚无Step6指纹。请核对最终JSON和实验配置后再复用。")
                if button_compat("复用现有Step5结果", key="reuse_legacy_step5"):
                    legacy_out = Path(s5.get("out_dir", str(STEP5_OUT)))
                    write_json(legacy_out / "step6_binding.json", solution_binding(json_path))
                    workflow_state["step5_out_dir"] = str(legacy_out.resolve())
                    save_state(workflow_state)
                    st.success("已记录人工核验并复用历史结果。")
            st.markdown("**默认实验设置**")
            metric_row([("分析模式", "No-LLM + τ扰动", "原Step5默认流程"), ("τ情景", "0.1 / 0.3 / 0.5", "统一扰动温度"), ("重复次数", 5, "每个LLM情景")])
            confirm = st.checkbox("确认运行Step5（包含LLM重复调用，可能产生API费用）", value=False, key="confirm_step5_cost")
            if s5["status"] != "verified" and button_compat("运行 / 更新 Step5", key="run_step5", type="primary", disabled=(not confirm or not step4_ready)):
                proxy_step1 = prepare_proxy_step1()
                with st.spinner("正在执行Step5消融与稳健性分析…"):
                    ok, out_dir, msg = run_step5(json_path, Path(workflow_state["active_summary_path"]), data_dir, review_dir, patent_file, proxy_step1, api_key, base_url, model, proxy, python_exe, repeats=5)
                if ok:
                    st.success(msg)
                    workflow_state["step5_done_hash"] = sha256_file(json_path)
                    workflow_state["step5_out_dir"] = str(out_dir.resolve())
                    save_state(workflow_state)
                else:
                    st.error(msg)
            if s5.get("out_dir"):
                out_dir = Path(s5["out_dir"])
            else:
                out_dir = STEP5_OUT
            if out_dir.exists():
                tables = out_dir / "tables"
                summary_csv = tables / "llm_analysis_summary_by_scenario.csv"
                if summary_csv.exists():
                    st.markdown("**实验结果汇总**")
                    dataframe_compat(read_csv_any(summary_csv), use_container_width=True, height=360)
                st.markdown("**实验图表**")
                figure_root = out_dir / "figures"
                if not figure_root.exists():
                    figure_root = out_dir / "llm_perturb_tau_0p3" / "figures"
                display_figures(figure_root, max_items=10)

# -----------------------------------------------------------------------------
# Footer: concise workflow state
# -----------------------------------------------------------------------------
with st.sidebar:
    divider_compat()
    state_now = load_state()
    solution_loaded = bool(state_now.get("active_solution_path"))
    solution_accepted = bool(state_now.get("solution_accepted"))
    pill_text = "优化方案已加载" if solution_loaded else "尚未加载优化方案"
    evidence_text = "证据已确认" if state_now.get("step1_confirmed") else "证据待确认"
    final_text = "最终方案已确认" if solution_accepted else "最终方案待评审"
    st.markdown(
        '<div class="paper-sidebar-status">'
        '<div class="paper-nav-title">DECISION STATUS</div>'
        f'<div class="paper-status-pill"><span class="paper-status-dot"></span>{html_lib.escape(pill_text)}</div>'
        f'<div class="paper-status-line">{html_lib.escape(evidence_text)}</div>'
        f'<div class="paper-status-line">{html_lib.escape(final_text)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
