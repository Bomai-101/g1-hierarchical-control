import runpy

from _legacy_bootstrap import PROJECT_ROOT, ensure_src_path

ensure_src_path()
runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_balance.py"), run_name="__main__")
