"""Verify release hashes; optionally verify the separately supplied model/data assets."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(base, entries):
    failures = []
    for entry in entries:
        path = (base / entry['path']).resolve()
        if path != base and base not in path.parents:
            failures.append((entry['path'], 'path leaves archive root'))
        elif not path.is_file():
            failures.append((entry['path'], 'missing'))
        elif path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            failures.append((entry['path'], 'size/hash mismatch'))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path)
    args = parser.parse_args()
    entries = json.loads((ROOT / 'manifests/package_files_sha256.json').read_text(encoding='utf-8'))
    failures = verify(ROOT.resolve(), entries)
    count = len(entries)
    if args.assets_dir:
        assets = json.loads((ROOT / 'manifests/reproduction_assets.json').read_text(encoding='utf-8'))
        failures.extend(verify(args.assets_dir.expanduser().resolve(), assets))
        count += len(assets)
    if failures:
        for path, why in failures:
            print('%s: %s' % (path, why))
        raise SystemExit(1)
    print('PASS: %d files match the release manifest.' % count)


if __name__ == '__main__':
    main()
