from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
ROOT = PAPER_DIR.parent

START_MARKER = "% === AUTO PARETO FIGURES START ==="
END_MARKER = "% === AUTO PARETO FIGURES END ==="

GENERATOR_SCRIPTS = [
    ROOT / "compare" / "plot_pareto_front.py",
    ROOT / "canshu" / "plot_pareto_front.py",
    ROOT / "xiaorong" / "plot_pareto_front.py",
]

FIGURE_INPUTS = [
    "../compare/pareto_figures.tex",
    "../canshu/pareto_figures.tex",
    "../xiaorong/pareto_figures.tex",
]


def find_tex_file(explicit_path: str | None) -> Path:
    if explicit_path:
        tex_file = Path(explicit_path)
        if not tex_file.is_absolute():
            tex_file = ROOT / tex_file
        return tex_file

    tex_files = sorted(PAPER_DIR.glob("*.tex"))
    if not tex_files:
        raise FileNotFoundError("No .tex file found in Paper.")

    preferred = [path for path in tex_files if path.name.lower().startswith("cgmoaco")]
    return preferred[0] if preferred else tex_files[0]


def run_generators() -> None:
    for script in GENERATOR_SCRIPTS:
        print(f"Run generator: {script}")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def package_exists(text: str, package: str) -> bool:
    for match in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", text):
        packages = [item.strip() for item in match.group(1).split(",")]
        if package in packages:
            return True
    return False


def ensure_package(text: str, package: str) -> str:
    if package_exists(text, package):
        return text

    matches = list(re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}", text))
    line = f"\\usepackage{{{package}}}"
    if matches:
        insert_at = matches[-1].end()
        return text[:insert_at] + "\n" + line + text[insert_at:]

    docclass = re.search(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", text)
    if docclass:
        insert_at = docclass.end()
        return text[:insert_at] + "\n" + line + text[insert_at:]

    return line + "\n" + text


def build_figure_block() -> str:
    lines = [
        START_MARKER,
        r"\clearpage",
        r"\section*{Pareto 前沿图}",
    ]
    for figure_input in FIGURE_INPUTS:
        lines.append(rf"\input{{{figure_input}}}")
    lines.append(END_MARKER)
    return "\n".join(lines)


def replace_or_insert_block(text: str, block: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start != -1 and end != -1 and end > start:
        end += len(END_MARKER)
        return text[:start] + block + text[end:]

    end_document = text.rfind(r"\end{document}")
    if end_document == -1:
        return text.rstrip() + "\n\n" + block + "\n"

    return text[:end_document].rstrip() + "\n\n" + block + "\n\n" + text[end_document:]


def compile_pdf(tex_file: Path) -> bool:
    command = ["xelatex", "-interaction=nonstopmode", tex_file.name]
    try:
        for _ in range(2):
            subprocess.run(command, cwd=tex_file.parent, check=True)
    except FileNotFoundError:
        print("xelatex not found. TeX file was updated, but PDF was not compiled.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"xelatex failed with exit code {exc.returncode}. Check the .log file.")
        return False
    return True


def update_pareto_figures(tex_file: Path, *, generate: bool, compile_: bool) -> None:
    if generate:
        run_generators()

    text = tex_file.read_text(encoding="utf-8")
    text = ensure_package(text, "graphicx")
    text = replace_or_insert_block(text, build_figure_block())
    tex_file.write_text(text, encoding="utf-8")
    print(f"Updated Pareto figure inputs in {tex_file}")

    if compile_:
        if compile_pdf(tex_file):
            print(f"PDF updated: {tex_file.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Pareto front figures in the paper PDF.")
    parser.add_argument("--tex", help="Path to the main TeX file. Defaults to the first Paper/*.tex.")
    parser.add_argument("--no-generate", action="store_true", help="Do not regenerate Pareto figures first.")
    parser.add_argument("--no-compile", action="store_true", help="Only update TeX; do not run xelatex.")
    args = parser.parse_args()

    tex_file = find_tex_file(args.tex)
    update_pareto_figures(
        tex_file,
        generate=not args.no_generate,
        compile_=not args.no_compile,
    )


if __name__ == "__main__":
    main()
