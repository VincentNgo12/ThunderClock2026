# File: licence_guard.py
# Purpose: PreToolUse hook for the mocap-curator agent. Blocks downloads from registration-gated
#          or third-party video hosts, installs into the simulator venv, and edits to files the
#          agent must never touch (user approvals, simulator ground truth, other subsystems).
# Contributors: Unassigned
"""Claude Code passes the pending tool call as JSON on stdin:
{"tool_name": "Bash" | "WebFetch" | "Edit" | "Write" | ..., "tool_input": {...}, ...}.
Exit 0 allows the call; exit 2 blocks it and shows stderr to the agent.
Standard library only, so it runs with any python3 (`python3 -I licence_guard.py`).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import PurePosixPath

# Class C (personal registration) and class D (third-party video) sources. The user downloads
# class C data by hand after accepting the licence; the agent never fetches it.
GATED_HOSTS = (
    "is.tue.mpg.de",                 # AMASS, BABEL, SMPL, SMPL-X, MANO
    "motion-database.humanoids.kit.edu",
    "dataverse.scholarsportal.info",  # BMLmovi
    "vision.imar.ro",                # Human3.6M
    "youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "instagram.com", "facebook.com",
)
GATED_TOOLS = ("yt-dlp", "youtube-dl", "gallery-dl")

# Paths relative to the repository root that the agent must not edit.
PROTECTED = (
    re.compile(r"(^|/)detection/simulation/motionlib/approvals/approved\.json$"),
    re.compile(r"(^|/)detection/simulation/thundersim/(motions|falls|spine|scenarios|device|transport|runner|"
               r"model_api|canonical|site|calibrate|baseline|tune|posture|report|__main__)\.py$"),
    re.compile(r"(^|/)detection/simulation/calibration\.json$"),
    re.compile(r"(^|/)detection/models/"),
    re.compile(r"(^|/)(firmware|backend|frontend|protocol)/"),
)

# pip/uv installs that target the simulator's runtime venv (offline tools use motionlib/.venv-tools).
VENV_INSTALL = re.compile(r"(\.venv/bin/(python|pip)[^\n;&|]*\bpip\b[^\n;&|]*\binstall\b"
                          r"|\.venv/bin/pip\d?\s+install"
                          r"|uv\s+pip\s+install\s+(?![^\n;&|]*\.venv-tools)[^\n;&|]*--python\s+\S*\.venv/bin/python)")


def block(reason: str) -> None:
    print(f"licence_guard: blocked. {reason}", file=sys.stderr)
    sys.exit(2)


def check_text(text: str) -> None:
    lowered = text.lower()
    for host in GATED_HOSTS:
        if host in lowered:
            block(f"'{host}' is a registration-gated or third-party video source. Write "
                  "motionlib/approvals/pending/<source>.json and stop with NEEDS_APPROVAL; the user downloads "
                  "class C data by hand, and class D video is refused.")
    for tool in GATED_TOOLS:
        if re.search(rf"\b{re.escape(tool)}\b", lowered):
            block(f"'{tool}' downloads third-party video, which this project does not use.")


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        block("could not parse the hook input, so the call is not allowed.")
    tool = event.get("tool_name", "")
    data = event.get("tool_input", {}) or {}
    if tool == "Bash":
        command = str(data.get("command", ""))
        check_text(command)
        if VENV_INSTALL.search(command):
            block("do not install packages into detection/simulation/.venv; offline tools go in "
                  "motionlib/.venv-tools (AGENTS.md: no dependency without a concrete need).")
    elif tool == "WebFetch":
        check_text(str(data.get("url", "")))
    elif tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = str(PurePosixPath(str(data.get("file_path", data.get("notebook_path", "")))))
        for pattern in PROTECTED:
            if pattern.search(path):
                block(f"{path} is outside what mocap-curator may edit (user approvals, simulator ground "
                      "truth, the detector or another subsystem).")
    sys.exit(0)


if __name__ == "__main__":
    main()
