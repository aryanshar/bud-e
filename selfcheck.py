"""Import and wiring check. Run after any edit, before running the pipeline.

    python selfcheck.py

Catches the class of bug where a module references a config attribute that
does not exist, which otherwise only surfaces when you speak to it.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent


def check_config_refs() -> list[str]:
    cfg = ast.parse((ROOT / "bud_e/config.py").read_text())
    defined = {t.id for n in cfg.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)}
    defined |= {n.name for n in cfg.body if isinstance(n, ast.FunctionDef)}

    problems = []
    files = [f for d in ("bud_e", "bench") for f in (ROOT / d).rglob("*.py")]
    for f in sorted(files):
        if f.name == "config.py":
            continue
        for name in re.findall(r"\bC\.([A-Za-z_][A-Za-z0-9_]*)",
                               f.read_text()):
            if name not in defined:
                problems.append(f"{f.relative_to(ROOT)}: C.{name} undefined")
    return problems


def check_imports() -> list[str]:
    problems = []
    for mod in ["bud_e.config", "bud_e.router", "bud_e.profile",
                "bud_e.llm", "bud_e.triggers", "bud_e.fast",
                "bud_e.asr", "bud_e.audio", "bud_e.speaker",
                "bud_e.ear", "bud_e.cli",
                "bench.cases", "bench.runner", "bench.report", "bench.cli"]:
        try:
            __import__(mod)
        except ImportError as e:
            problems.append(f"{mod}: {e}  (missing dependency?)")
        except Exception as e:
            problems.append(f"{mod}: {type(e).__name__}: {e}")
    return problems


def main() -> None:
    bad = check_config_refs()
    if bad:
        print("config references that do not resolve:")
        for b in bad:
            print(f"  {b}")
    else:
        print("config references  ok")

    bad2 = check_imports()
    if bad2:
        print("modules that fail to import:")
        for b in bad2:
            print(f"  {b}")
    else:
        print("module imports     ok")

    sys.exit(1 if (bad or bad2) else 0)


if __name__ == "__main__":
    main()
