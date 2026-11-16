#!/usr/bin/env python3
"""
Independent Baseline Integrity Verification Script
Audits all 25 frozen baseline files against stability_investigation/BASELINE_MANIFEST.json.
"""

import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone

def verify_baseline():
    script_path = Path(__file__).resolve()
    investigation_dir = script_path.parent.parent
    base_dir = investigation_dir.parent
    manifest_path = investigation_dir / "BASELINE_MANIFEST.json"
    audit_dir = investigation_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cert_path = audit_dir / "baseline_verification_certificate.json"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    core_files = manifest.get("core_files", {})
    total_files = len(core_files)
    
    print(f"================================================================================")
    print(f"INDEPENDENT BASELINE INTEGRITY AUDIT")
    print(f"Manifest: {manifest_path.name} | Total Files to Verify: {total_files}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"================================================================================\n")

    records = []
    all_matched = True

    print(f"{'Relative Path':<50} | {'Expected SHA-256':<16} | {'Actual SHA-256':<16} | {'Status'}")
    print("-" * 92)

    for rel_path, expected_meta in sorted(core_files.items()):
        file_path = base_dir / rel_path
        expected_hash = expected_meta["sha256"]
        expected_size = expected_meta["size_bytes"]

        if not file_path.exists():
            records.append({
                "path": rel_path,
                "exists": False,
                "expected_sha256": expected_hash,
                "actual_sha256": None,
                "size_bytes": 0,
                "status": "MISSING"
            })
            print(f"{rel_path:<50} | {expected_hash[:16]:<16} | {'MISSING':<16} | FAILED")
            all_matched = False
            continue

        file_bytes = file_path.read_bytes()
        actual_size = len(file_bytes)
        actual_hash = hashlib.sha256(file_bytes).hexdigest()
        hash_match = (actual_hash == expected_hash)
        size_match = (actual_size == expected_size)
        is_valid = hash_match and size_match

        status_str = "MATCH" if is_valid else "MISMATCH"
        if not is_valid:
            all_matched = False

        records.append({
            "path": rel_path,
            "exists": True,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "hash_match": hash_match,
            "size_match": size_match,
            "status": status_str
        })

        print(f"{rel_path:<50} | {expected_hash[:16]:<16} | {actual_hash[:16]:<16} | {status_str}")

    print("-" * 92)
    print(f"VERIFICATION SUMMARY: {sum(1 for r in records if r['status'] == 'MATCH')}/{total_files} files matched.")
    print(f"OVERALL INTEGRITY: {'PASSED (100% UNCHANGED)' if all_matched else 'FAILED'}")
    print(f"================================================================================\n")

    certificate = {
        "certificate_type": "BASELINE_IMMUTABILITY_AUDIT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_title": manifest.get("manifest_title", ""),
        "total_files_audited": total_files,
        "files_matched": sum(1 for r in records if r["status"] == "MATCH"),
        "all_matched": all_matched,
        "records": records
    }

    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2)

    print(f"Certificate written to: {cert_path}")
    
    if not all_matched:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    verify_baseline()
