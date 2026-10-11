"""Validate parsed views and summarize measured CPU/API/device costs.

Kernel sums may overlap streams; report the interval union separately. CPU
events are inclusive and must not be added across nesting as wall time.
"""
import argparse
import collections
import csv
import json
from pathlib import Path


def top(rows, name, value, count=None, limit=12):
    totals = collections.defaultdict(lambda: [0, 0.0])
    for row in rows:
        try:
            amount = float(str(row[value]).strip())
        except (ValueError, KeyError):
            continue
        totals[row[name]][0] += int(row[count]) if count else 1
        totals[row[name]][1] += amount
    return [{'name': key, 'count': values[0], 'total_us': values[1],
             'mean_us': values[1] / values[0]} for key, values in
            sorted(totals.items(), key=lambda x: x[1][1], reverse=True)[:limit]]


def union(intervals):
    total = 0.0
    start = end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + (end - start if end is not None else 0.0)


def summarize(path):
    out = path / 'ASCEND_PROFILER_OUTPUT'
    required = ['kernel_details.csv', 'op_statistic.csv', 'api_statistic.csv', 'trace_view.json']
    for name in required:
        if not (out / name).is_file():
            raise RuntimeError(f'missing {out / name}')
    with (out / 'kernel_details.csv').open() as f:
        reader = csv.DictReader(f)
        kernels = list(reader)
        columns = reader.fieldnames
    with (out / 'api_statistic.csv').open() as f:
        api = list(csv.DictReader(f))
    with (out / 'op_statistic.csv').open() as f:
        ops = list(csv.DictReader(f))
    trace = json.loads((out / 'trace_view.json').read_text())
    events = trace if isinstance(trace, list) else trace['traceEvents']
    if not kernels or not api or not ops or not events:
        raise RuntimeError(f'empty view in {path}')
    cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
    category_counts = collections.Counter(e.get('cat', '') for e in events)
    intervals = [(float(r['Start Time(us)']), float(r['Start Time(us)']) + float(r['Duration(us)']))
                 for r in kernels]
    return {
        'path': str(path), 'valid': True, 'kernel_rows': len(kernels), 'kernel_columns': columns,
        'devices': sorted(set(r['Device_id'] for r in kernels)),
        'streams': sorted(set(r['Stream ID'] for r in kernels)),
        'kernel_sum_us': sum(float(r['Duration(us)']) for r in kernels),
        'kernel_union_us': union(intervals),
        'kernel_span_us': max(b for a, b in intervals) - min(a for a, b in intervals),
        'trace_events': len(events), 'trace_categories': dict(category_counts),
        'top_kernel_types': top(kernels, 'Type', 'Duration(us)'),
        'top_kernel_names': top(kernels, 'Name', 'Duration(us)'),
        'top_api': top(api, 'API Name', 'Time(us)', 'Count'),
        'top_cpu_inclusive': top(cpu, 'name', 'dur'),
        'selected_cpu_inclusive': top([e for e in cpu if any(k in e.get('name', '').lower()
                      for k in ['index', 'nonzero', 'attention', 'synchronize', 'cumsum', 'copy', '_to'])],
                                     'name', 'dur', limit=30),
    }


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/ws/opt_20261011/profiles')
parser.add_argument('--modes', nargs='+', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
records = []
for mode in args.modes:
    paths = sorted((Path(args.root) / mode).glob('*_ascend_pt'))
    if not paths:
        raise RuntimeError(f'no worker profiles for {mode}')
    for path in paths:
        records.append(summarize(path))
Path(args.output).write_text(json.dumps(records, indent=2) + '\n')
for record in records:
    print(record['path'], 'kernel_sum_ms', round(record['kernel_sum_us']/1000, 3),
          'kernel_union_ms', round(record['kernel_union_us']/1000, 3),
          'kernel_span_ms', round(record['kernel_span_us']/1000, 3))
    for field in ['top_kernel_types', 'top_cpu_inclusive', 'selected_cpu_inclusive', 'top_api']:
        print(field, json.dumps(record[field][:8]))
