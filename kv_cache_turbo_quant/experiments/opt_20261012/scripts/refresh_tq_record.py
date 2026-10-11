#!/usr/bin/env python3
"""Refresh this provenanced record while preserving another session's cache edits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    record_id = 'tq-128k-benchmark-20261010'
    index = args.destination / 'index.json'
    marker = args.destination / '.memory-cache.json'
    before_index, before_marker = index.read_bytes(), marker.read_bytes()
    destination = json.loads(before_index)
    metadata = json.loads(before_marker)
    source = json.loads((args.source / 'index.json').read_bytes())
    selected = next(r for r in source['records'] if r['id'] == record_id)
    assert selected['visibility'] == 'shareable'
    previous = next(r for r in destination['records'] if r['id'] == record_id)
    assert previous['path'] == selected['path'] == 'projects/tq-128k-benchmark.md'
    document = args.destination / selected['path']
    before_document = document.read_bytes()
    assert digest(before_document) == metadata['files'][selected['path']], 'own cache record changed concurrently'
    source_document = (args.source / selected['path']).read_bytes()
    others = [r for r in destination['records'] if r['id'] != record_id]
    destination['records'] = [selected if r['id'] == record_id else r for r in destination['records']]
    assert others == [r for r in destination['records'] if r['id'] != record_id]
    now = datetime.now(timezone.utc).isoformat()
    destination['updated_at'] = now
    new_index = (json.dumps(destination, ensure_ascii=False, indent=2) + '\n').encode()
    metadata['files']['index.json'] = digest(new_index)
    metadata['files'][selected['path']] = digest(source_document)
    metadata.setdefault('partial_updates', []).append({'record_id': record_id,
        'source_commit': args.source_commit, 'updated_at': now,
        'note': 'Only this record and its index entry refreshed; unrelated local cache edits preserved. source_commit at marker top remains the base snapshot.'})
    assert index.read_bytes() == before_index and marker.read_bytes() == before_marker and document.read_bytes() == before_document, 'cache changed concurrently'
    document.write_bytes(source_document)
    index.write_bytes(new_index)
    marker.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'record_id': record_id, 'source_commit': args.source_commit,
                      'cache_refresh': 'record only; other local modifications preserved'}))


if __name__ == '__main__':
    main()
