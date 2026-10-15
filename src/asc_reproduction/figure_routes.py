"""Dispatch figure numbers to existing CLIs, without classifier training.

Locations come from explicit arguments or a local batch plan. Original artwork
is explicitly retained, never mislabeled as regenerated scientific output.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .provenance import file_hash, write_json


ROOT = Path(__file__).resolve().parents[2]

# Module names invoke existing CLIs; .py paths invoke existing scripts.
# Additional arguments pin the approved mode, not new analysis parameters.
ROUTES = {
    "Figure_01": {"retain": ("retain",)},
    "Figure_02": {
        "render-a": ("driver_oncoplot",), "render-b": ("ccf_panels",),
        "assemble": ("scripts/assemble_figures/figure_02.py",),
    },
    "Figure_03": {"render": ("figure3_composite", "--prepared")},
    "Figure_04": {
        "render-a": ("cgc_oncoplot",), "render-bc": ("subtype_metrics",),
        "render-d": ("subtype_c47_profile",),
        "assemble": ("scripts/assemble_figures/figure_04.py",),
    },
    "Figure_05": {
        "render-ab": ("pca_panels", "--prepared"),
        "render-cd": ("softscore_figures",),
        "assemble": ("scripts/assemble_figures/figure_05.py",),
    },
    "Supplementary_Figure_01": {"render": ("clustering_metrics",)},
    "Supplementary_Figure_02": {"retain": ("retain",)},
    "Supplementary_Figure_03": {"retain": ("retain",)},
    "Supplementary_Figure_04": {"retain": ("retain",)},
    "Supplementary_Figure_05": {
        "render": ("scripts/render_s5.py",),
        "assemble": ("scripts/assemble_figures/supplementary_05.py",),
    },
    "Supplementary_Figure_06": {
        "render-ab": ("cnv_heatmaps", "--prepared"),
        "render-cd": ("pca_panels", "--prepared"),
        "assemble": ("scripts/assemble_figures/supplementary_06.py",),
    },
    "Supplementary_Figure_07": {
        "render": ("similarity_panels", "--s7-prepared"),
        "assemble": ("scripts/assemble_figures/supplementary_07.py",),
    },
    "Supplementary_Figure_08": {
        "render": ("component_metrics",),
        "assemble": ("scripts/assemble_figures/supplementary_08.py",),
    },
    "Supplementary_Figure_09": {
        "render": ("paired_screen",),
        "assemble": ("scripts/assemble_figures/supplementary_09.py",),
    },
    "Supplementary_Figure_10": {
        "render": ("s10_figures",),
        "assemble": ("scripts/assemble_figures/supplementary_10.py",),
    },
    "Supplementary_Figure_11": {"render": ("subtype_oncoplot",)},
    "Supplementary_Figure_12": {
        "render": ("protein_lollipops",),
        "assemble": ("scripts/assemble_figures/supplementary_12.py",),
    },
    "Supplementary_Figure_13": {"render": ("smoking_oncoplot",)},
    "Supplementary_Figure_14": {
        "render": ("subtype_signatures",),
        "assemble": ("scripts/assemble_figures/supplementary_14.py",),
    },
    "Supplementary_Figure_15": {
        "statistics": ("subtype_cnv",), "render": ("scripts/render_s15.py",),
        "assemble": ("scripts/assemble_figures/supplementary_15.py",),
    },
    "Supplementary_Figure_16": {
        "render": ("scripts/render_s16.py",),
        "assemble": ("scripts/assemble_figures/supplementary_16.py",),
    },
    "Supplementary_Figure_17": {
        "render": ("scripts/render_s17.py",),
        "assemble": ("scripts/assemble_figures/supplementary_17.py",),
    },
    "Supplementary_Figure_18": {
        "render-a": ("pooled_cca", "--square"),
        "render-b": ("reference_cca", "--merged"),
        "assemble": ("scripts/assemble_figures/supplementary_18.py",),
    },
    "Supplementary_Figure_19": {
        "render": ("model_s19_saved",),
        "assemble": ("scripts/assemble_figures/supplementary_19.py",),
    },
    "Supplementary_Figure_20": {
        "render": ("scripts/assemble_figures/supplementary_20.py",),
    },
}


def command_for(figure, step, arguments):
    """Resolve only registered figure commands, never arbitrary executables."""
    target, *flags = ROUTES[figure][step]
    if target == "retain":
        return None
    if target.endswith(".py"):
        command = [sys.executable, str(ROOT / target)]
    else:
        command = [sys.executable, "-m", f"asc_reproduction.{target}"]
    return command + list(arguments) + flags


def output_path(arguments):
    """Require an explicit new output location for every executable step."""
    if arguments.count("--output") != 1:
        raise ValueError("Each step requires exactly one explicit --output")
    index = arguments.index("--output")
    if index + 1 == len(arguments):
        raise ValueError("Missing --output value")
    path = Path(arguments[index + 1]).resolve()
    if path.exists():
        raise FileExistsError(f"Refuse existing output: {path}")
    return path


def retain_assets(arguments):
    """Copy specified artwork with hashes, without claiming regeneration."""
    parser = argparse.ArgumentParser(description=retain_assets.__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    names = [source.name for source in args.sources]
    if len(set(names)) != len(names) or "retained_assets.json" in names:
        raise ValueError("Retained asset filenames collide")
    hashes = {source.name: file_hash(source) for source in args.sources}
    args.output.mkdir(parents=True, exist_ok=False)
    for source in args.sources:
        target = args.output / source.name
        shutil.copy2(source, target)
        if file_hash(target) != hashes[source.name]:
            raise ValueError("Retained asset copy differs")
    write_json(args.output / "retained_assets.json", {
        "status": "retained artwork; not regenerated", "sha256": hashes,
    })


def validate_jobs(jobs, *, all_figures=False):
    """Reject missing routes and conflicting outputs before launching a batch."""
    seen = set()
    outputs = []
    for job in jobs:
        figure, step, arguments = job["figure"], job["step"], job["args"]
        if not isinstance(arguments, list) or not all(
            isinstance(value, str) for value in arguments
        ):
            raise ValueError("Job args must be a list of strings")
        if figure not in ROUTES or step not in ROUTES[figure]:
            raise ValueError(f"Unknown figure route: {figure}/{step}")
        key = (figure, step)
        if key in seen:
            raise ValueError(f"Duplicate figure step: {key}")
        seen.add(key)
        output = output_path(arguments)
        if any(output == other or output in other.parents or other in output.parents
               for other in outputs):
            raise ValueError("Output directories overlap")
        outputs.append(output)
    expected = {(figure, step) for figure, steps in ROUTES.items() for step in steps}
    if all_figures and seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Incomplete all-figure plan; missing: {missing}")


def run_job(job, *, dry_run=False):
    """Dispatch existing code and check that the requested output was created."""
    arguments = job["args"]
    command = command_for(job["figure"], job["step"], arguments)
    if dry_run:
        print(shlex.join(command) if command else f"retain artwork: {arguments}")
        return
    output = output_path(arguments)
    if command is None:
        retain_assets(arguments)
    else:
        subprocess.run(command, check=True)
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        raise RuntimeError(f"Figure step produced no output: {output}")


def main(argv=None):
    """List all 25 figure routes or execute an explicit step/batch plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--figure", choices=[*ROUTES, "all"])
    parser.add_argument("--step")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(ROUTES, indent=2))
        return
    if args.plan:
        if args.step or args.arguments:
            parser.error("A plan cannot be combined with a step or forwarded args")
        jobs = json.loads(args.plan.read_text())["jobs"]
        if args.figure not in (None, "all"):
            jobs = [job for job in jobs if job["figure"] == args.figure]
        if not jobs:
            parser.error("No jobs selected")
        validate_jobs(jobs, all_figures=args.figure == "all")
    else:
        if args.figure in (None, "all") or not args.step:
            parser.error("Use --figure/--step, or --figure all --plan PLAN.json")
        arguments = args.arguments
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        job = {"figure": args.figure, "step": args.step, "args": arguments}
        if arguments == ["--help"]:
            command = command_for(args.figure, args.step, arguments)
            if command is None:
                retain_assets(arguments)
            else:
                subprocess.run(command, check=True)
            return
        jobs = [job]
        validate_jobs(jobs)
    for job in jobs:
        run_job(job, dry_run=args.dry_run)
