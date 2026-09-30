from _legacy_bootstrap import ensure_src_path

ensure_src_path()
from g1_control.evaluation.robustness import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
