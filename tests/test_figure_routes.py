"""Route coverage/no-overwrite checks without study analysis or model fitting."""

import pytest

from asc_reproduction import figure_routes as routes


def test_catalog_has_exact_route_coverage():
    """Every manuscript figure is explicit, including retained schematics."""
    expected = {f"Figure_{number:02d}" for number in range(1, 6)}
    expected.update(
        f"Supplementary_Figure_{number:02d}" for number in range(1, 21)
    )
    assert set(routes.ROUTES) == expected
    assert len(routes.ROUTES) == 25
    for steps in routes.ROUTES.values():
        for target, *flags in steps.values():
            if target == "retain":
                continue
            source = (routes.ROOT / target if target.endswith(".py") else
                      routes.ROOT / "src/asc_reproduction" / f"{target}.py")
            assert source.is_file()
            assert not any(name in target for name in (
                "model_runner", "model_finalize", "model_s19_composite"
            ))


def test_all_requires_complete_plan_before_dispatch(tmp_path):
    """A partial batch cannot silently claim all figures were reproduced."""
    job = {"figure": "Supplementary_Figure_19", "step": "render",
           "args": ["--output", str(tmp_path / "new")]}
    with pytest.raises(ValueError, match="Incomplete"):
        routes.validate_jobs([job], all_figures=True)
    jobs = [
        {"figure": figure, "step": step,
         "args": ["--output", str(tmp_path / figure / step)]}
        for figure, steps in routes.ROUTES.items() for step in steps
    ]
    routes.validate_jobs(jobs, all_figures=True)


def test_s18_s19_canonical_modes():
    """Only merged A/B S18 and four-row S19 entries are dispatched."""
    figure = "Supplementary_Figure_18"
    assert routes.command_for(figure, "render-b", [])[-1] == "--merged"
    assert routes.command_for(figure, "render-a", [])[-1] == "--square"
    command = routes.command_for("Supplementary_Figure_19", "render", [])
    assert "asc_reproduction.model_s19_saved" in command


def test_retained_asset_is_identical_and_explicit(tmp_path):
    """Original artwork is copied, not relabeled as regenerated analysis."""
    source = tmp_path / "source.svg"
    source.write_text("<svg/>")
    output = tmp_path / "retained"
    routes.run_job({"figure": "Figure_01", "step": "retain",
                    "args": ["--sources", str(source), "--output", str(output)]})
    assert (output / source.name).read_bytes() == source.read_bytes()
    assert "not regenerated" in (output / "retained_assets.json").read_text()
    with pytest.raises(FileExistsError):
        routes.output_path(["--output", str(output)])


def test_overlapping_outputs_rejected(tmp_path):
    """A later assembly cannot overwrite an earlier panel directory."""
    jobs = [{"figure": "Supplementary_Figure_19", "step": step,
             "args": ["--output", str(tmp_path / path)]}
            for step, path in [("render", "new"), ("assemble", "new/nested")]]
    with pytest.raises(ValueError, match="overlap"):
        routes.validate_jobs(jobs)
