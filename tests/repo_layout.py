"""Which repository layout the tests run in.

Most tests only need this project directory.  A few also read things that exist in only one layout:

* "development": this directory sits inside the development repository, whose root holds
  ``control/evidence/upstream/accepted-assets.json`` (the upstream asset register that STATE-X01 and
  EVAL-X01 read) and the CI workflow under ``.github/workflows/``.
* "standalone": this directory is itself the repository root; the workflow is
  ``.github/workflows/queryshield-ci.yml`` inside it, and there is no register.
* "unknown": neither.

The checks below look at what is actually there, not at how many directories up the repository root is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
WORKFLOW_RELATIVE = Path(".github") / "workflows" / "queryshield-ci.yml"
REGISTER_RELATIVE = Path("control") / "evidence" / "upstream" / "accepted-assets.json"
STANDALONE_WORKFLOW = PROJECT_ROOT / WORKFLOW_RELATIVE
REGISTER = REPO_ROOT / REGISTER_RELATIVE


def layout() -> str:
    if STANDALONE_WORKFLOW.is_file():
        return "standalone"
    if REGISTER.is_file():
        return "development"
    return "unknown"


REGISTER_SKIP_REASON = (
    "needs the upstream asset register (control/evidence/upstream/accepted-assets.json), "
    "which exists only in the development repository"
)
needs_register = pytest.mark.skipif(layout() != "development", reason=REGISTER_SKIP_REASON)
