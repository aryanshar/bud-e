import argparse

from bud_e import config as C


def main() -> None:
    p = argparse.ArgumentParser(prog="bud-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark one model")
    r.add_argument("--model", default=C.LOCAL_MODEL)
    r.add_argument("--base", default=None)
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--think", action="store_true",
                   help="leave reasoning on (default: /no_think)")

    sub.add_parser("models", help="list what the endpoint has loaded")
    sub.add_parser("report", help="compare every saved run")

    a = p.parse_args()

    if a.cmd == "run":
        from bench.runner import run
        run(a.model, runs=a.runs, base=a.base, save=not a.no_save,
            think=a.think)
    elif a.cmd == "models":
        from bud_e.llm import list_models
        for m in list_models():
            print(m)
    elif a.cmd == "report":
        from bench.report import report
        report()


if __name__ == "__main__":
    main()
