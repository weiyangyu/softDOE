import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cholesky

from softDOE import soft
from softDOE import _core
from softDOE._soft import _q_min_value, _qmc_generate
from r_helpers import local_soft_prelude


N_SIMULATIONS = 30
SMALL_DESIGN_M = 500
DESIGN_MS = (500, 1000)
EVAL_M = 500


def _require_r_package(package):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not available.")

    check = subprocess.run(
        [
            "Rscript",
            "-e",
            f"quit(status = ifelse(requireNamespace('{package}', quietly = TRUE), 0, 1))",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        pytest.skip(f"R package {package} is not available.")


def _read_matrix(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _save_svg_scatter(design, path, title):
    width = 420
    height = 420
    pad = 36
    plot_size = width - 2 * pad

    points = []
    for x, y in design[:, :2]:
        sx = pad + x * plot_size
        sy = height - pad - y * plot_size
        points.append(f'<circle cx="{sx:.3f}" cy="{sy:.3f}" r="4" fill="#1f77b4" />')

    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white" />',
                f'<text x="{pad}" y="22" font-family="sans-serif" font-size="14">{title}</text>',
                f'<rect x="{pad}" y="{pad}" width="{plot_size}" height="{plot_size}" fill="none" stroke="black" />',
                *points,
                "</svg>",
            ]
        )
        + "\n"
    )


def _save_svg_q_distribution(py_values, r_values, path, title):
    py_values = np.asarray(py_values, dtype=float)
    r_values = np.asarray(r_values, dtype=float)
    py_values = np.log(py_values)
    r_values = np.log(r_values)
    all_values = np.concatenate([py_values, r_values])

    width = 620
    height = 420
    pad_left = 58
    pad_right = 24
    pad_top = 42
    pad_bottom = 54
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 11)
    if np.allclose(bins[0], bins[-1]):
        center = bins[0]
        bins = np.linspace(center - 1.0, center + 1.0, 11)

    py_counts, _ = np.histogram(py_values, bins=bins)
    r_counts, _ = np.histogram(r_values, bins=bins)
    max_count = max(int(np.max(py_counts)), int(np.max(r_counts)), 1)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{pad_left}" y="24" font-family="sans-serif" font-size="14">{title}</text>',
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="black" />',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="black" />',
        f'<text x="{pad_left}" y="{height - 14}" font-family="sans-serif" font-size="11">log(Q objective value)</text>',
        f'<text x="8" y="{pad_top + 12}" font-family="sans-serif" font-size="11">count</text>',
    ]

    bin_width_px = plot_width / len(py_counts)
    bar_width = bin_width_px * 0.82
    x_offset = (bin_width_px - bar_width) / 2

    for idx, (r_count, py_count) in enumerate(zip(r_counts, py_counts)):
        x = pad_left + idx * bin_width_px + x_offset
        r_height = plot_height * (r_count / max_count)
        py_height = plot_height * (py_count / max_count)
        baseline = height - pad_bottom

        elements.append(
            f'<rect x="{x:.3f}" y="{baseline - r_height:.3f}" width="{bar_width:.3f}" height="{r_height:.3f}" fill="#ff7f0e" fill-opacity="0.45" />'
        )
        elements.append(
            f'<rect x="{x:.3f}" y="{baseline - py_height:.3f}" width="{bar_width:.3f}" height="{py_height:.3f}" fill="#1f77b4" fill-opacity="0.45" />'
        )

    for tick in np.linspace(0, max_count, 5):
        y = height - pad_bottom - plot_height * (tick / max_count)
        elements.append(f'<line x1="{pad_left - 4}" y1="{y:.3f}" x2="{pad_left}" y2="{y:.3f}" stroke="black" />')
        elements.append(
            f'<text x="{pad_left - 8}" y="{y + 4:.3f}" text-anchor="end" font-family="sans-serif" font-size="10">{tick:.0f}</text>'
        )

    for idx in [0, len(bins) // 2, len(bins) - 1]:
        x = pad_left + plot_width * ((bins[idx] - bins[0]) / (bins[-1] - bins[0]))
        elements.append(f'<line x1="{x:.3f}" y1="{height - pad_bottom}" x2="{x:.3f}" y2="{height - pad_bottom + 4}" stroke="black" />')
        elements.append(
            f'<text x="{x:.3f}" y="{height - pad_bottom + 18}" text-anchor="middle" font-family="sans-serif" font-size="10">{bins[idx]:.2e}</text>'
        )

    legend_x = width - pad_right - 130
    elements.extend(
        [
            f'<rect x="{legend_x}" y="{pad_top}" width="12" height="12" fill="#1f77b4" fill-opacity="0.45" />',
            f'<text x="{legend_x + 18}" y="{pad_top + 11}" font-family="sans-serif" font-size="11">Python</text>',
            f'<rect x="{legend_x}" y="{pad_top + 18}" width="12" height="12" fill="#ff7f0e" fill-opacity="0.45" />',
            f'<text x="{legend_x + 18}" y="{pad_top + 29}" font-family="sans-serif" font-size="11">R</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(elements) + "\n")


def _assert_ofat_structure(design, p, l):
    blocks = [design[i * l : (i + 1) * l] for i in range(p + 1)]
    changed = []
    for idx, block in enumerate(blocks[1:], start=1):
        for previous in blocks[:idx]:
            diffs = [not np.allclose(block[:, j], previous[:, j]) for j in range(p)]
            if sum(diffs) == 1:
                changed.append(diffs.index(True))
                break
        else:
            pytest.fail("Could not find a prior OFAT block that differs in one factor.")
    assert sorted(changed) == list(range(p))


def _q_objective(design, qmc):
    p = design.shape[1]
    l = design.shape[0] // (p + 1)
    beta = 1.0 / ((2 * l) ** 2)
    R = _core.corr_matrix(design, None, None, 1.0, beta)
    L = cholesky(R + np.eye(R.shape[0]) * 1e-6, lower=True, check_finite=False)
    R_qmc = _core.q_matrix(qmc, design, None, None, 1.0, beta)
    return _q_min_value(R_qmc, L)


def _assert_q_distributions_comparable(py_values, r_values):
    py_values = np.asarray(py_values, dtype=float)
    r_values = np.asarray(r_values, dtype=float)

    assert py_values.shape == r_values.shape == (N_SIMULATIONS,)
    assert np.all(np.isfinite(py_values))
    assert np.all(np.isfinite(r_values))

    py_mean = float(np.mean(py_values))
    r_mean = float(np.mean(r_values))
    relative_gap = abs(py_mean - r_mean) / max(abs(r_mean), 1e-12)

    assert relative_gap < 0.35


def test_soft_matches_r_soft_by_plots_structure_and_q_values(tmp_path, capsys):
    _require_r_package("Rcpp")
    _require_r_package("RcppArmadillo")
    _require_r_package("SFDesign")
    _require_r_package("gtools")
    _require_r_package("purrr")

    script_path = tmp_path / "soft_reference.R"
    script_path.write_text(
        textwrap.dedent(
            """
            __LOCAL_SOFT_PRELUDE__

            set.seed(1)
            standard_small <- soft(2, 4, structure = "standard", m = 500)
            strict_small <- soft(2, 4, structure = "strict", m = 500)

            write.csv(standard_small, "r_standard_small.csv", row.names = FALSE)
            write.csv(strict_small, "r_strict_small.csv", row.names = FALSE)

            for (design_m in c(__DESIGN_MS__)) {
              for (i in seq_len(__N_SIMULATIONS__)) {
                set.seed(1000 + i)
                standard_design <- soft(10, 4, structure = "standard", m = design_m)
                strict_design <- soft(10, 4, structure = "strict", m = design_m)

                write.csv(standard_design, sprintf("r_standard_m%d_%02d.csv", design_m, i), row.names = FALSE)
                write.csv(strict_design, sprintf("r_strict_m%d_%02d.csv", design_m, i), row.names = FALSE)
              }
            }
            """
        )
        .replace("__LOCAL_SOFT_PRELUDE__", local_soft_prelude())
        .replace("__N_SIMULATIONS__", str(N_SIMULATIONS))
        .replace("__DESIGN_MS__", ", ".join(str(value) for value in DESIGN_MS))
    )

    subprocess.run(["Rscript", str(script_path)], cwd=tmp_path, check=True)

    py_standard_small = soft(2, 4, structure="standard", m=SMALL_DESIGN_M, seed=1)
    py_strict_small = soft(2, 4, structure="strict", m=SMALL_DESIGN_M, seed=1)
    r_standard_small = _read_matrix(tmp_path / "r_standard_small.csv")
    r_strict_small = _read_matrix(tmp_path / "r_strict_small.csv")

    plot_dir = Path(__file__).resolve().parent / "plots" / "test_2_2"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _save_svg_scatter(py_standard_small, plot_dir / "python_standard.svg", "Python standard SOFT")
    _save_svg_scatter(r_standard_small, plot_dir / "r_standard.svg", "R standard SOFT")
    _save_svg_scatter(py_strict_small, plot_dir / "python_strict.svg", "Python strict SOFT")
    _save_svg_scatter(r_strict_small, plot_dir / "r_strict.svg", "R strict SOFT")

    _assert_ofat_structure(py_standard_small, p=2, l=4)
    _assert_ofat_structure(r_standard_small, p=2, l=4)
    _assert_ofat_structure(py_strict_small, p=2, l=4)
    _assert_ofat_structure(r_strict_small, p=2, l=4)

    qmc = _qmc_generate(EVAL_M, 10, np.random.default_rng(20240615))
    q_results = {}

    for design_m in DESIGN_MS:
        py_standard_q = []
        py_strict_q = []
        r_standard_q = []
        r_strict_q = []

        for i in range(1, N_SIMULATIONS + 1):
            py_standard_design = soft(
                10, 4, structure="standard", m=design_m, seed=1000 + i
            )
            py_strict_design = soft(
                10, 4, structure="strict", m=design_m, seed=1000 + i
            )
            r_standard_design = _read_matrix(
                tmp_path / f"r_standard_m{design_m}_{i:02d}.csv"
            )
            r_strict_design = _read_matrix(
                tmp_path / f"r_strict_m{design_m}_{i:02d}.csv"
            )

            _assert_ofat_structure(py_standard_design, p=10, l=4)
            _assert_ofat_structure(py_strict_design, p=10, l=4)
            _assert_ofat_structure(r_standard_design, p=10, l=4)
            _assert_ofat_structure(r_strict_design, p=10, l=4)

            py_standard_q.append(_q_objective(py_standard_design, qmc))
            py_strict_q.append(_q_objective(py_strict_design, qmc))
            r_standard_q.append(_q_objective(r_standard_design, qmc))
            r_strict_q.append(_q_objective(r_strict_design, qmc))

        q_results[design_m] = {
            "py_standard": py_standard_q,
            "r_standard": r_standard_q,
            "py_strict": py_strict_q,
            "r_strict": r_strict_q,
        }

        _save_svg_q_distribution(
            py_standard_q,
            r_standard_q,
            plot_dir / f"q_distribution_standard_m{design_m}.svg",
            f"Standard SOFT log(Q) distribution, m={design_m}",
        )
        _save_svg_q_distribution(
            py_strict_q,
            r_strict_q,
            plot_dir / f"q_distribution_strict_m{design_m}.svg",
            f"Strict SOFT log(Q) distribution, m={design_m}",
        )

    _save_svg_q_distribution(
        q_results[500]["py_standard"],
        q_results[500]["r_standard"],
        plot_dir / "q_distribution_standard.svg",
        "Standard SOFT log(Q) distribution, m=500",
    )
    _save_svg_q_distribution(
        q_results[500]["py_strict"],
        q_results[500]["r_strict"],
        plot_dir / "q_distribution_strict.svg",
        "Strict SOFT log(Q) distribution, m=500",
    )

    with capsys.disabled():
        print(f"\nSaved 2D diagnostic plots to: {plot_dir}")
        for design_m, values in q_results.items():
            print(f"\nDesign-search m = {design_m}")
            print("Standard Python Q values:")
            print(np.array(values["py_standard"]))
            print("Standard R Q values:")
            print(np.array(values["r_standard"]))
            print("Strict Python Q values:")
            print(np.array(values["py_strict"]))
            print("Strict R Q values:")
            print(np.array(values["r_strict"]))

    for values in q_results.values():
        _assert_q_distributions_comparable(values["py_standard"], values["r_standard"])
        _assert_q_distributions_comparable(values["py_strict"], values["r_strict"])
