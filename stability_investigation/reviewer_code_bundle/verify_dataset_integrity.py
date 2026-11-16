"""
Dataset Cryptographic Integrity Verifier for Reviewer Reproduction.
Verifies:
1. benchmark_manifest.json matches recorded SHA-256 digest: 8d5b02c591989e49a168a662727dd51ede02f6778dde643aca7c3d584e316342
2. All 36 generated synthetic graph instances match their recorded SHA-256 digests.
"""

import sys
import json
import hashlib
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = "8d5b02c591989e49a168a662727dd51ede02f6778dde643aca7c3d584e316342"

EXPECTED_INSTANCE_HASHES = {
    "bottleneck/test_inst_0_seed_311.pt": "d8cb9697db76feec2e25164e4f6e840aafdca3a07fc9f4051eb1ef3065ba4cf7",
    "bottleneck/test_inst_1_seed_312.pt": "4eb6163472eddc21eb3b9ac5fef556923189eea44d4b5356f46456cbd4d1dd35",
    "bottleneck/test_inst_2_seed_313.pt": "129fb00e6721a8f117bbcb4cd9a7b5a105a077d2aad814f0d08ed627db1e6f31",
    "bottleneck/test_inst_3_seed_314.pt": "388f1bf39ee94aaa3b5b316f223f2905cf89bde17fa3bf086f8c4832d571d3d5",
    "bottleneck/test_inst_4_seed_315.pt": "55a8542f388a16d6651e6a41d199523f1dd8884c297a71f0a7a2ab4a5416a59f",
    "bottleneck/train_inst_0_seed_111.pt": "b57d60bbec619f7c02d9e64fb64accb148c2997d28c6626988989234140f15ae",
    "bottleneck/train_inst_1_seed_112.pt": "a5cdc164a24c892430b60de08e0975cc719f9b74e1edd6fb1b8e6be906ac733c",
    "bottleneck/train_inst_2_seed_113.pt": "dc65b8b6fbac71c48a794389aae9038a223c6dce993a2ffed96e19abb40a70d3",
    "bottleneck/train_inst_3_seed_114.pt": "1115ddee3957143666150f25b79ce8088fdecd1704ff38c64bbb1318765d78e1",
    "bottleneck/train_inst_4_seed_115.pt": "f67c1d1a2dbc7508ac76fd2c7080249490ab4583ecd824edb24e6c434f39cd6d",
    "bottleneck/val_inst_0_seed_211.pt": "74c20515fea5a601c40826ea689c2a3dc24a7e3d6167dd54e2a6de80d6427391",
    "bottleneck/val_inst_1_seed_212.pt": "a2606f4ea2611cddb050d09f32c875d1df9bbd32a6f75faa0f052f5c6b656671",
    "chain_ladder/test_inst_0_seed_301.pt": "4fbe1f4ddcf8478b1b7c7651e63e9c7fc9175f76bec730715c34fa49e5a91d17",
    "chain_ladder/test_inst_1_seed_302.pt": "bdd5c6b307621d0810c2e405cd1ccac182a19e02f88bebfc33df103f0bb98919",
    "chain_ladder/test_inst_2_seed_303.pt": "d11b8a68e2448d81ee077d5530d882be29ad8235b23825f1c58ac50bb2ea5a0c",
    "chain_ladder/test_inst_3_seed_304.pt": "e6882f58cb3fc0687cec7f768d15d29df7c2dfb6dae5d50dd476173d82876252",
    "chain_ladder/test_inst_4_seed_305.pt": "6b32384e2092a6587c78e1efe78daa038c58c280cf44169317c18e1de9c9fce0",
    "chain_ladder/train_inst_0_seed_101.pt": "daa3876eb6d9f4ac88d4815c190c6c067395a70b32d32f47f7785e908462f288",
    "chain_ladder/train_inst_1_seed_102.pt": "377e25f94e3d49f1eacefa34a6892e7d00faa4ed6b5dbad5cc47f707a80171fa",
    "chain_ladder/train_inst_2_seed_103.pt": "9ce8bfcb91a6d22f9259777fe397e99289948bf2396f21383fb1a85cc9ee1839",
    "chain_ladder/train_inst_3_seed_104.pt": "c83f069d07494201b45ac9e47c74aac09e41d8ba8ed52890c0c42e008b8ebba5",
    "chain_ladder/train_inst_4_seed_105.pt": "8175072e6f9c9ba436ba61e1a9df0f4c9d037244644ced7d4dfeaa3086be41b7",
    "chain_ladder/val_inst_0_seed_201.pt": "bfb2e30f5a1e1fb8a770bef5f5d367150aa83b26b03a7f926b488a10162ae422",
    "chain_ladder/val_inst_1_seed_202.pt": "ef4e6983a4969a4e47484170b7ef0b72999dbccafe42636b8c168a3caaef68a4",
    "random_dag/test_inst_0_seed_321.pt": "6a881561356c3e4dd1abc2e6c099e4fbc237641ee9a9f9e2aee9ac2d3f697206",
    "random_dag/test_inst_1_seed_322.pt": "4f7d8f9ff82ac1e1fc418dd158c6381006333e4b16efca9c4d75b969769fdaa3",
    "random_dag/test_inst_2_seed_323.pt": "de7eb0ece23e1d908b51ca6f1e87b2efe1bb159b96e7734000d9814332d2430d",
    "random_dag/test_inst_3_seed_324.pt": "a238c2e305a7d8ad044efea90c18bc483c95566be45c1db40e114f434125b847",
    "random_dag/test_inst_4_seed_325.pt": "4e629207b73c4dff1c23dcbd897327d1d159067c38e3bebd8bb434827a26e716",
    "random_dag/train_inst_0_seed_121.pt": "48f68fdfa8cd016be69a510ffd2266cabc605306670409f81a7a6ee2b5a9f56e",
    "random_dag/train_inst_1_seed_122.pt": "06e14e90b61e614a0e4ae88450bd68eab117e9c345a2ec6c36e42311d1376dad",
    "random_dag/train_inst_2_seed_123.pt": "917d6519abc6973e5f2a8365c72d446f449bcdc2f034c5ffe7631e1cea24c530",
    "random_dag/train_inst_3_seed_124.pt": "5ea5b63fde26432e32aef2feb9c4b75e41c42cbb095ae601d319bc8e0e14e859",
    "random_dag/train_inst_4_seed_125.pt": "d5b9ecf0367a2b9b55626ec579dc6066672f53fe164ef70ac6e22dc3cf849426",
    "random_dag/val_inst_0_seed_221.pt": "33d3600a2734cd1c0c531e5c49c850ca940e79a1bdbc667a097bc92e1da9e079",
    "random_dag/val_inst_1_seed_222.pt": "a2449db5b9f3c30f1597355678e641532b670263dcdbc885dafee8a2a3c3487d"
}

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_dataset():
    root = Path(__file__).resolve().parent
    manifest_p = root / "benchmark_manifest.json"
    if not manifest_p.exists():
        manifest_p = root / "data" / "synthetic" / "benchmark_manifest.json"
    if not manifest_p.exists():
        print(f"ERROR: Cannot find benchmark_manifest.json at {manifest_p}. Please run python3 dataset_generator.py first.")
        sys.exit(1)

    actual_manifest_hash = sha256_file(manifest_p)
    print("=" * 80)
    print("SYNTHETIC BENCHMARK CRYPTOGRAPHIC INTEGRITY AUDIT")
    print(f"Manifest File: {manifest_p.name}")
    print(f"Expected SHA-256: {EXPECTED_MANIFEST_SHA256}")
    print(f"Actual SHA-256:   {actual_manifest_hash}")
    print("=" * 80)

    if actual_manifest_hash != EXPECTED_MANIFEST_SHA256:
        print("ERROR: Manifest SHA-256 digest does not match expected research release digest!")
        sys.exit(1)

    with open(manifest_p) as f:
        manifest = json.load(f)

    data_dir = root / "data" / "synthetic"
    all_match = True
    match_count = 0
    total_files = len(EXPECTED_INSTANCE_HASHES)

    header_file = "Relative File Path"
    header_hash = "Expected Digest"
    header_status = "Status"
    print(f"{header_file:<45} | {header_hash:<16} | {header_status}")
    print("-" * 80)

    for rel_path, exp_hash in sorted(EXPECTED_INSTANCE_HASHES.items()):
        fp = data_dir / rel_path
        if not fp.exists():
            print(f"{rel_path:<45} | {exp_hash[:16]}... | MISSING")
            all_match = False
            continue
        act_hash = sha256_file(fp)
        if act_hash != exp_hash:
            print(f"{rel_path:<45} | {exp_hash[:16]}... | MISMATCH ({act_hash[:16]}...)")
            all_match = False
        else:
            print(f"{rel_path:<45} | {act_hash[:16]}... | MATCH")
            match_count += 1

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {match_count}/{total_files} graph instance files matched.")
    if all_match and match_count == total_files:
        print("OVERALL DATASET INTEGRITY: PASSED (100% BYTE-FOR-BYTE MATCH)")
        print("=" * 80)
    else:
        print("OVERALL DATASET INTEGRITY: FAILED")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    verify_dataset()
