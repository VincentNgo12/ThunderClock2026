# File: acquire.py
# Purpose: Stage S1 of the motion library: download the class A files listed in sources.json,
#          one folder per source under motionlib/raw/<source>/, and record or verify a sha256
#          for every file. Can also list a remote zip and fetch single members by HTTP range.
# Contributors: Unassigned
"""Standard library only. Run from detection/simulation with the isolated interpreter:

    .venv/bin/python -I motionlib/tools/acquire.py fetch [--source cmu]   # download what is missing
    .venv/bin/python -I motionlib/tools/acquire.py verify                 # re-hash every file on disk
    .venv/bin/python -I motionlib/tools/acquire.py list-zip <url>         # print a remote zip's members

sources.json lists every file by explicit URL. An entry with "member" is one file inside a remote
zip: only that member's bytes are fetched (HTTP Range), its CRC-32 is checked against the zip's
central directory, and the sha256 is of the extracted member. Downloads are data only: nothing
downloaded is executed or imported, and archives are never extracted to disk as a tree.
Only class A sources are fetched; other classes need the user to place files by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path, PurePosixPath

MOTIONLIB = Path(__file__).resolve().parents[1]
SOURCES = MOTIONLIB / "sources.json"
USER_AGENT = "ThunderSim-motionlib/1.0 (offline research fetch)"
TIMEOUT_S = 120


def http_get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """GET a URL (optionally bytes first..last inclusive). Follows redirects."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        data = response.read()
        if byte_range is not None and response.status != 206:
            raise RuntimeError(f"{url}: server ignored the Range header (status {response.status})")
    return data


def remote_size(url: str) -> int:
    """Total size from a 1-byte range request (Content-Range: bytes 0-0/<size>)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        content_range = response.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError(f"{url}: no Content-Range, cannot read the zip directory remotely")
    return int(content_range.rsplit("/", 1)[1])


def zip_directory(url: str) -> dict[str, dict]:
    """Central directory of a remote (non-zip64) zip: member name -> offset, sizes, crc, method."""
    size = remote_size(url)
    tail_len = min(size, 65536 + 22)
    tail = http_get(url, (size - tail_len, size - 1))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError(f"{url}: end-of-central-directory record not found")
    _, _, _, _, count, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[eocd:eocd + 22])
    if cd_offset == 0xFFFFFFFF or count == 0xFFFF:
        raise RuntimeError(f"{url}: zip64 archives are not supported by this reader")
    directory = http_get(url, (cd_offset, cd_offset + cd_size - 1))
    members = {}
    pos = 0
    for _ in range(count):
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", directory[pos:pos + 46])
        if fields[0] != 0x02014B50:
            raise RuntimeError(f"{url}: bad central directory entry at {pos}")
        method, crc, comp_size, raw_size = fields[4], fields[7], fields[8], fields[9]
        name_len, extra_len, comment_len, local_offset = fields[10], fields[11], fields[12], fields[16]
        name = directory[pos + 46:pos + 46 + name_len].decode("utf-8", "replace")
        members[name] = {"method": method, "crc32": crc, "comp_size": comp_size, "size": raw_size,
                         "offset": local_offset}
        pos += 46 + name_len + extra_len + comment_len
    return members


def fetch_member(url: str, info: dict) -> bytes:
    """One zip member via two range requests (local header, then data). Stored or deflated only."""
    header = http_get(url, (info["offset"], info["offset"] + 29))
    signature, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50:
        raise RuntimeError(f"{url}: bad local header at {info['offset']}")
    start = info["offset"] + 30 + name_len + extra_len
    packed = http_get(url, (start, start + info["comp_size"] - 1)) if info["comp_size"] else b""
    if info["method"] == 0:
        data = packed
    elif info["method"] == 8:
        data = zlib.decompress(packed, -15)
    else:
        raise RuntimeError(f"{url}: compression method {info['method']} is not supported")
    if len(data) != info["size"] or (zlib.crc32(data) & 0xFFFFFFFF) != info["crc32"]:
        raise RuntimeError(f"{url}: member size or CRC-32 mismatch")
    return data


def safe_target(source: str, rel_path: str) -> Path:
    """raw/<source>/<file>; refuses absolute paths and '..' so a listing cannot escape the folder."""
    rel = PurePosixPath(rel_path)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 3 or rel.parts[0] != "raw" or rel.parts[1] != source:
        raise RuntimeError(f"path {rel_path!r} must be raw/{source}/<file>")
    return MOTIONLIB.joinpath(*rel.parts)


def load_sources() -> dict:
    return json.loads(SOURCES.read_text())


def save_sources(sources: dict) -> None:
    SOURCES.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n")


def fetch(only: str | None) -> int:
    sources = load_sources()
    failures = 0
    zip_cache: dict[str, dict] = {}
    for key, source in sources["sources"].items():
        if only and key != only:
            continue
        if source.get("class") != "A":
            print(f"[{key}] class {source.get('class')}: not fetched (needs files placed by the user)")
            continue
        for entry in source.get("files", []):
            target = safe_target(key, entry["path"])
            if target.exists():
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if entry.get("sha256") and digest != entry["sha256"]:
                    print(f"[{key}] {entry['path']}: sha256 MISMATCH on disk")
                    failures += 1
                elif not entry.get("sha256"):
                    entry["sha256"], entry["bytes"] = digest, target.stat().st_size
                continue
            try:
                if "member" in entry:
                    if entry["url"] not in zip_cache:
                        zip_cache[entry["url"]] = zip_directory(entry["url"])
                    info = zip_cache[entry["url"]].get(entry["member"])
                    if info is None:
                        raise RuntimeError(f"member {entry['member']} not in the zip")
                    data = fetch_member(entry["url"], info)
                    entry["crc32"] = f"{info['crc32']:08x}"
                else:
                    data = http_get(entry["url"])
                    if entry.get("md5") and hashlib.md5(data).hexdigest() != entry["md5"]:
                        raise RuntimeError("md5 does not match the publisher's checksum")
            except Exception as exc:  # report and carry on with the other files
                print(f"[{key}] {entry['path']}: FAILED ({exc})")
                failures += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            if entry.get("sha256") and digest != entry["sha256"]:
                print(f"[{key}] {entry['path']}: sha256 MISMATCH, not saved")
                failures += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entry["sha256"], entry["bytes"] = digest, len(data)
            print(f"[{key}] {entry['path']}: {len(data)} bytes, sha256 {digest[:16]}...")
        save_sources(sources)
    return failures


def verify() -> int:
    failures = 0
    for key, source in load_sources()["sources"].items():
        for entry in source.get("files", []):
            target = safe_target(key, entry["path"])
            if not target.exists():
                print(f"[{key}] {entry['path']}: missing")
                failures += 1
            elif hashlib.sha256(target.read_bytes()).hexdigest() != entry.get("sha256"):
                print(f"[{key}] {entry['path']}: sha256 mismatch")
                failures += 1
    print("verify:", "OK" if failures == 0 else f"{failures} problem(s)")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--source")
    sub.add_parser("verify")
    p_list = sub.add_parser("list-zip")
    p_list.add_argument("url")
    args = parser.parse_args()
    if args.command == "fetch":
        sys.exit(1 if fetch(args.source) else 0)
    if args.command == "verify":
        sys.exit(1 if verify() else 0)
    for name, info in sorted(zip_directory(args.url).items()):
        print(f"{info['size']:>11} {name}")


if __name__ == "__main__":
    main()
