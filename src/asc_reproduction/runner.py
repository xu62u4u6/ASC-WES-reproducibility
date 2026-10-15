"""Run implemented reproduction stages with verified resume and resource records.

Linux is required for flock, resource accounting and /proc memory sampling.
Scientific decisions still pending in TODO.md remain pending after a successful run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import runpy
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

# Compatibility exports for existing callers.
from .provenance import file_hash, runtime_identity, utc_now, write_json

SNV = "data/WES/all_case_table.db"
GENE = "data/CNV_0407/gene_cnv.db"
ARM = "data/CNV_0407/arm_cnv.db"
CLUSTERS = "data/clustering/best_k_cluster_labels.csv"
CASES = "data/ASC_all_case_metadata_0312.csv"
# stage: module, inputs actually read, expected TIFF count
STAGES = {
    "figure3-composite": ("figure3_composite", (SNV, ARM, GENE, CLUSTERS), 1),
    "model-workflow": ("model_workflow", (), 1),
    "clustering-metrics": (
        "clustering_metrics",
        ("data/clustering/metric_df.csv", CLUSTERS),
        1,
    ),
    "paired-3q": ("paired_cnv", (SNV, ARM), 2),
    "paired-screen": ("paired_screen", (SNV, ARM, GENE, CLUSTERS), 4),
    "subtype-cnv": ("subtype_cnv", (SNV, ARM, GENE, CLUSTERS), 2),
    "c47-profile": ("c47_profile", (SNV, GENE, CLUSTERS), 1),
    "paired-c47": ("paired_c47_difference", (SNV, GENE, CLUSTERS), 1),
    "subtype-c47": ("subtype_c47_profile", (SNV, GENE, CLUSTERS), 1),
    "subtype-metrics": ("subtype_metrics", (SNV,), 2),
    "cgc-oncoplot": (
        "cgc_oncoplot",
        (SNV, "data/Census All Months Jan 27 2025.csv"),
        1,
    ),
    "cluster-panels": ("cluster_panels", (SNV, GENE, CLUSTERS), 5),
    "cnv-heatmaps": ("cnv_heatmaps", (SNV, GENE, ARM, CLUSTERS, CASES), 3),
    "pca-panels": ("pca_panels", (SNV, GENE, ARM, CLUSTERS, CASES), 4),
    "pooled-cca": ("pooled_cca", (SNV, GENE, CLUSTERS), 1),
    "similarity-panels": ("similarity_panels", (SNV, GENE, ARM, CLUSTERS, CASES), 8),
    "subtype-oncoplot": ("subtype_oncoplot", (SNV,), 1),
    "smoking-oncoplot": ("smoking_oncoplot", (SNV,), 1),
    "component-metrics": ("component_metrics", (SNV,), 4),
    "subtype-signatures": (
        "subtype_signatures",
        (SNV, "data/signature/signature_table.db"),
        6,
    ),
    "paired-oncoplot": ("paired_oncoplot", (SNV, CASES), 1),
    "ccf-panels": ("ccf_panels", (CASES, "data/pyclone_output/*.pyclone.tsv"), 1),
    "driver-oncoplot": (
        "driver_oncoplot",
        (SNV, CASES, "data/WES/all_case_maf_AST.csv"),
        1,
    ),
    "protein-lollipops": (
        "protein_lollipops",
        (SNV, "data/WES/LUAD.maf", "data/WES/ASC.maf", "data/WES/LUSC.maf"),
        2,
    ),
}
# Prepared products must be fingerprinted even for prepared-only renderers.
PREPARED_STAGE_INPUTS = {
    "paired-oncoplot": ("paired_SNV.h5",),
    "figure3-composite": ("cohort.h5",),
    "cnv-heatmaps": (
        "tumor_CNV_gene.h5",
        "paired_CNV_arm.h5",
        "paired_CNV_cluster.h5",
    ),
    "pca-panels": (
        "tumor_SNV.h5",
        "paired_SNV.h5",
        "tumor_CNV_gene.h5",
        "paired_CNV_arm.h5",
        "paired_CNV_cluster.h5",
    ),
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
TABLES = {
    "figure3-composite": ("C47_gene_level_component_means.csv",),
    "clustering-metrics": ("S1_saved_fold_summary.csv",),
    "protein-lollipops": ("S12A_position_counts.csv", "S12B_position_counts.csv"),
    "paired-3q": ("3q_gain_summary_ge_0.10.csv",),
    "paired-screen": ("paired_arm_statistics.csv", "paired_cluster_statistics.csv"),
    "subtype-cnv": (
        "arm_all_tests.csv",
        "cluster_all_tests.csv",
        "Supplementary_Table_1.csv",
        "Supplementary_Table_2_candidate_legacy_order.csv",
        "Supplementary_Table_2_candidate_stable_feature_order.csv",
        "Supplementary_Table_2_candidate_include_cutoff_ties.csv",
        "Supplementary_Table_2_tie_audit.json",
    ),
    "c47-profile": ("C47_gene_level_component_means.csv",),
    "subtype-c47": ("C47_gene_level_subtype_means.csv",),
    "subtype-metrics": (
        "Figure4B_C_group_summary.csv",
        "Figure4B_C_pairwise_raw_statistics.csv",
    ),
    "cgc-oncoplot": ("Figure_4A_gene_summary.csv",),
    "cluster-panels": ("pairwise_tests.csv", "group_summary.csv"),
    "similarity-panels": ("similarity_summary.csv",),
    "subtype-oncoplot": ("S11_gene_summary.csv",),
    "smoking-oncoplot": ("S13_gene_summary.csv",),
    "component-metrics": (
        "S8_group_summary.csv",
        "S8_pairwise_raw.csv",
        "S8_friedman_raw.csv",
    ),
    "subtype-signatures": ("S14_group_summary.csv", "S14_pairwise_raw.csv"),
    "paired-oncoplot": ("S5A_gene_summary.csv",),
    "driver-oncoplot": ("Figure2A_variant_summary.csv",),
    "pca-panels": tuple(
        f"{panel}_loadings.csv"
        for panel in (
            "Figure_5A_SNV_PCA",
            "Figure_5B_CNV_PCA",
            "S6C_SNV_PCA",
            "S6D_CNV_PCA",
        )
    ),
    "pooled-cca": (
        "Supplementary_Table_3.csv",
        "Supplementary_Table_4.csv",
        "pooled_CCA_snv_weights_all.csv",
        "pooled_CCA_cnv_weights_all.csv",
        "CNV_standardization_parameters.csv",
    ),
}


def input_identity(data_root: Path, stage: str, domain_csv: Path | None = None) -> dict:
    identity = {}
    if stage == "protein-lollipops":
        if domain_csv is None or not domain_csv.is_file():
            raise ValueError("protein-lollipops requires an external domain CSV")
        identity["external:protein_domains"] = {
            "bytes": domain_csv.stat().st_size,
            "sha256": file_hash(domain_csv),
        }
    for name in STAGES[stage][1]:
        if "*" not in name:
            identity[name] = {
                "bytes": (data_root / name).stat().st_size,
                "sha256": file_hash(data_root / name),
            }
            continue
        paths = sorted(data_root.glob(name))
        if not paths or any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("Missing or unsafe multi-file input")
        # Filename/content association matters, but patient-bearing filenames
        # must never appear in the serialized provenance record.
        members = [
            {
                "name": str(p.relative_to(data_root)),
                "bytes": p.stat().st_size,
                "sha256": file_hash(p),
            }
            for p in paths
        ]
        digest = hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        identity[name] = {
            "files": len(members),
            "bytes": sum(m["bytes"] for m in members),
            "sha256": digest,
            "scope": "ordered filename/content inventory; names not exported",
        }
    if stage in PREPARED_STAGE_INPUTS:
        from .cohort_inputs import validate_prepared_inputs

        identity.update(
            validate_prepared_inputs(data_root, PREPARED_STAGE_INPUTS[stage])
        )
    return identity


def process_tree_rss_kib(pid: int) -> int:
    """Sample summed RSS; shared pages may be counted in multiple processes."""
    rows = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            try:
                fields = (item / "stat").read_text().rsplit(")", 1)[1].split()
                rows[int(item.name)] = (int(fields[1]), int(fields[21]))
            except (OSError, ValueError, IndexError):
                continue
    descendants = {pid}
    while True:
        added = {
            child for child, (parent, _) in rows.items() if parent in descendants
        } - descendants
        if not added:
            break
        descendants.update(added)
    pages = sum(rows.get(child, (0, 0))[1] for child in descendants)
    return pages * os.sysconf("SC_PAGE_SIZE") // 1024


def worker(
    stage: str,
    data_root: Path,
    output: Path,
    report: Path,
    domain_csv: Path | None = None,
) -> int:
    """Run one legacy-compatible stage and retain diagnostics in its local log."""
    started = time.monotonic()
    record = {"stage": stage, "started_at": utc_now(), "status": "running"}
    write_json(report, record)
    code = 0
    try:
        module = "asc_reproduction." + STAGES[stage][0]
        sys.argv = [module, "--data-root", str(data_root), "--output", str(output)]
        if stage in PREPARED_STAGE_INPUTS and stage != "paired-oncoplot":
            sys.argv.append("--prepared")
        if stage == "protein-lollipops":
            if domain_csv is None:
                raise ValueError("Missing external domain CSV")
            sys.argv += ["--domain-csv", str(domain_csv)]
        runpy.run_module(module, run_name="__main__")
    except SystemExit as error:
        code = (
            error.code
            if isinstance(error.code, int)
            else (0 if error.code is None else 1)
        )
        if code:
            record["error_type"] = "SystemExit"
    except BaseException as error:
        code = 1
        traceback.print_exc()
        # Raw exception messages can contain sample keys; keep only code locations.
        record["error_type"] = type(error).__name__
        record["trace"] = [
            {
                "file": Path(frame.filename).name,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(error.__traceback__)
        ]
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(
        status="succeeded" if code == 0 else "failed",
        completed_at=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        exit_code=code,
        worker_peak_rss_kib=usage.ru_maxrss,
        reaped_children_peak_rss_kib=children.ru_maxrss,
        cpu_seconds=usage.ru_utime
        + usage.ru_stime
        + children.ru_utime
        + children.ru_stime,
    )
    write_json(report, record)
    return code


def execute_stage(
    stage: str,
    data_root: Path,
    output: Path,
    report: Path,
    progress: Path,
    threads: int,
    lock_fd: int,
    domain_csv: Path | None = None,
) -> dict:
    env = os.environ.copy()
    env.update({name: str(threads) for name in THREAD_VARIABLES})
    started = time.monotonic()
    peak = 0
    command = [
        sys.executable,
        "-m",
        "asc_reproduction.runner",
        "__worker",
        stage,
        str(data_root),
        str(output),
        str(report),
    ]
    if domain_csv is not None:
        command.append(str(domain_csv))
    # Full diagnostics stay in a private local log, outside exported summaries.
    # These logs may contain sample identifiers and must not be published.
    log_path = report.with_suffix(".log")
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(log_fd, "w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )
    try:
        while process.poll() is None:
            current = process_tree_rss_kib(process.pid)
            peak = max(peak, current)
            write_json(
                progress,
                {
                    "stage": stage,
                    "status": "running",
                    "pid": process.pid,
                    "elapsed_seconds": time.monotonic() - started,
                    "sampled_tree_rss_kib": current,
                    "sampled_tree_peak_rss_kib": peak,
                },
            )
            time.sleep(0.5)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    info = (
        json.loads(report.read_text())
        if report.is_file()
        else {"status": "missing_worker_report"}
    )
    info.update(
        local_log=log_path.name,
        process_exit_code=process.returncode,
        sampled_tree_peak_rss_kib=peak,
        parent_elapsed_seconds=time.monotonic() - started,
    )
    if process.returncode != 0 or info.get("status") != "succeeded":
        info["status"] = "failed"
    write_json(progress, {"stage": stage, **info})
    return info


def artifact_inventory(
    folder: Path, expected_tiffs: int, required_tables: tuple = ()
) -> dict:
    from PIL import Image

    summary = json.loads((folder / "summary.json").read_text())
    if not isinstance(summary, dict) or not summary:
        raise ValueError("Stage summary is missing or empty")
    for name in required_tables:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Required aggregate table or audit report is missing")
    tiffs = list(folder.glob("*.tiff"))
    if len(tiffs) != expected_tiffs:
        raise ValueError("Unexpected TIFF count")
    for path in tiffs:
        with Image.open(path) as im:
            im.load()
            if im.mode != "RGB" or tuple(
                round(float(d)) for d in im.info.get("dpi", ())
            ) != (600, 600):
                raise ValueError("TIFF must be decoded RGB at 600 dpi")
        preview = path.with_suffix(".png")
        with Image.open(preview) as im:
            im.load()
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in files):
        raise ValueError("Artifacts must be regular files")
    return {
        str(path.relative_to(folder)): {
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        for path in files
    }


def reusable_manifest(root: Path, stage: str, identity: dict) -> dict | None:
    latest = root / f"{stage}.latest.json"
    if not latest.is_file():
        return None
    try:
        manifest = json.loads(latest.read_text())
        folder = (root / manifest["artifact_directory"]).resolve()
        if (
            root.resolve() not in folder.parents
            or manifest.get("status") != "succeeded"
            or manifest.get("identity") != identity
        ):
            return None
        inventory = artifact_inventory(folder, STAGES[stage][2], TABLES.get(stage, ()))
        if inventory != manifest.get("artifacts"):
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another runner is using this output directory"
            ) from None
        marker = root / ".asc_runner.json"
        if not marker.exists():
            if any(p.name != ".runner.lock" for p in root.iterdir()):
                raise ValueError(
                    "Existing non-runner output directory must not be overwritten"
                )
            write_json(marker, {"schema": 1, "created_at": utc_now()})
        # Worker inherits this descriptor so a parent crash cannot permit a
        # second runner to start while the first worker still holds the lock.
        yield lock.fileno()


def run_stages(
    stages: list[str],
    data_root: Path,
    root: Path,
    *,
    resume: bool,
    threads: int,
    domain_csv: Path | None = None,
) -> dict:
    if threads < 1:
        raise ValueError("threads must be positive")
    if "protein-lollipops" in stages and (
        domain_csv is None or not domain_csv.is_file()
    ):
        raise ValueError("protein-lollipops requires --domain-csv")

    def inputs(stage):
        return (
            input_identity(data_root, stage, domain_csv)
            if stage == "protein-lollipops"
            else input_identity(data_root, stage)
        )

    runtime = runtime_identity()
    invocation = {
        "started_at": utc_now(),
        "scope": "implemented stages only; TODO includes unfinished analyses and scientific decisions",
        "status": "running",
        "stages": [],
    }
    with output_lock(root) as lock_fd:
        write_json(root / "run_summary.json", invocation)
        try:
            for stage in stages:
                identity = {
                    "stage": stage,
                    "runtime": runtime,
                    "inputs": inputs(stage),
                    "threads": threads,
                }
                cached = reusable_manifest(root, stage, identity) if resume else None
                if cached and (
                    inputs(stage) != identity["inputs"] or runtime_identity() != runtime
                ):
                    raise RuntimeError(
                        "Inputs or source changed during resume verification"
                    )
                if cached:
                    invocation["stages"].append(
                        {
                            "stage": stage,
                            "status": "verified_cached",
                            "artifact_directory": cached["artifact_directory"],
                        }
                    )
                    print(f"{stage}: verified cached artifacts", flush=True)
                    write_json(root / "run_summary.json", invocation)
                    continue
                attempt = f"{stage}-{uuid4().hex[:12]}"
                folder = root / attempt
                folder.mkdir()
                record = {
                    "stage": stage,
                    "artifact_directory": attempt,
                    "identity": identity,
                    "status": "running",
                }
                write_json(root / f"{attempt}.manifest.json", record)
                write_json(root / f"{stage}.latest.json", record)
                print(f"{stage}: started", flush=True)
                try:
                    extra = (
                        {"domain_csv": domain_csv}
                        if stage == "protein-lollipops"
                        else {}
                    )
                    execution = execute_stage(
                        stage,
                        data_root,
                        folder,
                        root / f"{attempt}.execution.json",
                        root / "progress.json",
                        threads,
                        lock_fd,
                        **extra,
                    )
                    record["execution"] = execution
                    if execution["status"] != "succeeded":
                        raise RuntimeError(
                            "Worker failed; inspect structured execution report"
                        )
                    # Rehash inputs after execution: a changed input cannot receive a success marker.
                    if (
                        inputs(stage) != identity["inputs"]
                        or runtime_identity() != runtime
                    ):
                        raise RuntimeError("Inputs or source changed during execution")
                    record["artifacts"] = artifact_inventory(
                        folder, STAGES[stage][2], TABLES.get(stage, ())
                    )
                    record["status"] = "succeeded"
                except BaseException as error:
                    record.update(
                        status="failed", verification_error_type=type(error).__name__
                    )
                    raise
                finally:
                    write_json(root / f"{attempt}.manifest.json", record)
                    write_json(root / f"{stage}.latest.json", record)
                    invocation["stages"].append(
                        {
                            "stage": stage,
                            "status": record["status"],
                            "artifact_directory": attempt,
                        }
                    )
                    write_json(root / "run_summary.json", invocation)
                print(
                    f"{stage}: succeeded; {execution['elapsed_seconds']:.1f}s; worker peak {execution['worker_peak_rss_kib'] / 1024:.1f} MiB",
                    flush=True,
                )
            invocation["status"] = "succeeded"
        except BaseException as error:
            invocation.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            invocation["completed_at"] = utc_now()
            write_json(root / "run_summary.json", invocation)
    return invocation


def main() -> int:
    if sys.platform != "linux":
        raise RuntimeError("This runner currently requires Linux")
    if len(sys.argv) > 1 and sys.argv[1] == "__worker":
        _, _, stage, data, output, report, *extra = sys.argv
        if len(extra) > 1:
            raise ValueError("Invalid worker arguments")
        return worker(
            stage,
            Path(data),
            Path(output),
            Path(report),
            Path(extra[0]) if extra else None,
        )
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stages", choices=list(STAGES), nargs="+")
    selection.add_argument(
        "--all-available",
        action="store_true",
        help="all implemented stages, not the entire manuscript",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--domain-csv",
        type=Path,
        help="external protein annotation, required for protein-lollipops and --all-available",
    )
    args = parser.parse_args()
    try:
        run_stages(
            list(STAGES) if args.all_available else list(dict.fromkeys(args.stages)),
            args.data_root.resolve(),
            args.output.resolve(),
            resume=args.resume,
            threads=args.threads,
            domain_csv=args.domain_csv.resolve() if args.domain_csv else None,
        )
    except Exception as error:
        # Keep raw exception text (which may include identifiers) out of output logs.
        print(
            f"Runner failed ({type(error).__name__}); inspect run_summary.json and execution records.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
