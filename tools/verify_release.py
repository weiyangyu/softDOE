"""Build and test wheel/sdist installs in isolated environments without editable imports.

Run from any directory after installing build and twine:
    python tools/verify_release.py
Use --run-r for version-checked live MOFAT comparisons as well.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "release-checks",
                        help="Parent directory for a new, uniquely named verification run.")
    parser.add_argument("--run-r", action="store_true")
    parser.add_argument("--mofat-version", default="2.0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.output.resolve()))
    inputs = [ROOT / name for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "CITATION.cff", "LICENSE")]
    inputs += [path for path in (ROOT / "src" / "softDOE").iterdir()
               if path.suffix in {".py", ".cpp", ".hpp", ".mm", ".metal", ".cu", ".npy", ".txt"}]
    report = {"started": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "source": str(ROOT), "commands": [], "artifacts": {}, "status": "running",
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(inputs)}}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_CACHE_DIR"] = "1"
    # Keep small numerical tests portable and avoid nested BLAS oversubscription.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"

    def save():
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, cwd):
        print(f"[{label}] {' '.join(map(str, command))}", flush=True)
        started = time.monotonic()
        log = run_dir / f"{label}.log"
        with log.open("w") as out:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=out, stderr=subprocess.STDOUT)
        report["commands"].append({"label": label, "command": list(map(str, command)),
                                   "cwd": str(cwd), "exit_code": result.returncode,
                                   "seconds": time.monotonic()-started, "log": str(log)})
        save()
        lines = log.read_text(errors="replace").splitlines()
        print("\n".join(lines[-(40 if result.returncode else 3):]), flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {log}")

    print(f"Verification output: {run_dir}", flush=True)
    save()
    try:
        artifacts = run_dir / "artifacts"
        # Build a fresh source snapshot, with no existing egg-info or build
        # outputs and no concurrent writes to the development checkout.
        source = run_dir / "source"
        for path in inputs:
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        run("build", [sys.executable,"-m","build","--outdir",artifacts,source], source)
        wheels = list(artifacts.glob("*.whl"))
        archives = list(artifacts.glob("*.tar.gz"))
        if len(wheels) != 1 or len(archives) != 1:
            raise RuntimeError("Expected one wheel and one source archive")
        run("metadata", [sys.executable,"-m","twine","check","--strict",*archives,*wheels], ROOT)
        for artifact in [*archives, *wheels]:
            with (zipfile.ZipFile(artifact) if artifact.suffix == ".whl" else tarfile.open(artifact)) as package:
                names = package.namelist() if artifact.suffix == ".whl" else package.getnames()
            for suffix in ("softDOE/_sobol_directions.npy", "softDOE/THIRD_PARTY_NOTICES.txt",
                           "softDOE/_strict.metal", "softDOE/_strict.cu"):
                if not any(name.endswith(suffix) for name in names):
                    raise RuntimeError(f"Missing runtime asset {suffix} in {artifact}")
            forbidden = {".venv",".work","__pycache__","tests","benchmarks","build","dist"}
            if any(forbidden.intersection(Path(name).parts) for name in names):
                raise RuntimeError(f"Development files leaked into {artifact}")
            if artifact.suffix != ".whl":
                for suffix in ("_core.cpp","_native.cpp","_native.hpp","_kernel.hpp",
                               "_native_metal.mm","_native_cuda.cpp","pyproject.toml","setup.py"):
                    if not any(name.endswith("/"+suffix) for name in names):
                        raise RuntimeError(f"Missing build input {suffix}")
                if any(name.endswith((".so",".pyd",".pyc")) for name in names):
                    raise RuntimeError("Local compiled files leaked into source archive")
            report["artifacts"][artifact.name] = {
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "files": names}
        save()
        # Separate test directories prevent imports from src/ or an editable install.
        for label, artifact in (("wheel",wheels[0]),("sdist",archives[0])):
            isolated = run_dir / label
            isolated.mkdir()
            environment = isolated / "venv"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            shutil.copytree(ROOT / "tests", isolated / "tests", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "examples", isolated / "examples", ignore=shutil.ignore_patterns("__pycache__"))
            run(label+"-install", [python,"-m","pip","install",artifact,"pytest"], isolated)
            smoke = (
                "import importlib.metadata as m,json,pathlib,sys,softDOE; "
                "p=pathlib.Path(softDOE.__file__).resolve(); "
                "assert pathlib.Path(sys.prefix).resolve() in p.parents, str(p); "
                "assert softDOE.__version__==m.version('softDOE'); "
                "print(json.dumps({'path':str(p),'version':softDOE.__version__,"
                "'dependencies':{n:m.version(n) for n in ['numpy','scipy','pydoe']},"
                "'backends':softDOE.backend_info()},default=str))"
            )
            run(label+"-import", [python,"-c",smoke], isolated)
            run(label+"-dependencies", [python,"-m","pip","check"], isolated)
            command = [python,"-m","pytest","-q","-ra","tests"]
            if args.run_r:
                command += ["--run-r", "--mofat-version="+args.mofat_version]
            run(label+"-tests", command, isolated)
            run(label+"-example", [python,"examples/mofat_examples.py"], isolated)
            readme_blocks = []
            import re
            for match in re.finditer(r"```python\n(.*?)```",(ROOT/"README.md").read_text(),re.S):
                readme_blocks.append(match.group(1))
            example = isolated / "readme_examples.py"
            example.write_text("\n\n".join(readme_blocks))
            run(label+"-readme", [python,example], isolated)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["finished"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"Report: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
