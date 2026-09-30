import runpy

from _legacy_bootstrap import PROJECT_ROOT, ensure_src_path

ensure_src_path()
runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_controller_smoke.py"), run_name="__main__")
