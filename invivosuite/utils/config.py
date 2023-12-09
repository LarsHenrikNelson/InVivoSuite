from pathlib import Path


def check_dir():
    p = Path.home()
    h = ".invivosuite"
    prog_dir = Path(p / h)
    if not prog_dir.exists():
        prog_dir.mkdir()
    return prog_dir


PROG_DIR = check_dir()
