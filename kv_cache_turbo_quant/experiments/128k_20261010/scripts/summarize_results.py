#!/usr/bin/env python3
"""Validate full-work cohorts and produce a paired result table."""
import argparse
import hashlib
import json
import math
import pathlib


def validate(path):
    data = json.loads(path.read_text())
    assert data['input_len'] == 131072 and data['output_len'] == 1024, path
    assert data['num_prompts'] == data['concurrency'], path
    assert len(data['requests']) == data['num_prompts'], path
    assert len(data['payload_sha256']) == data['num_prompts'], path
    successes = [r for r in data['requests'] if r['status'] == 'success']
    assert len(successes) == data['successful'], path
    assert sum(r['status'] == 'failed' for r in data['requests']) == data['failed'], path
    for request in successes:
        assert request['usage']['prompt_tokens'] == 131072, path
        assert request['usage']['completion_tokens'] == 1024, path
    if data['complete_cohort']:
        assert data['successful'] == data['num_prompts'] and data['failed'] == 0, path
        expected = data['num_prompts'] * 1024 / data['duration_s']
        assert math.isclose(data['output_throughput_tps'], expected, rel_tol=1e-9), path
    else:
        assert data['output_throughput_tps'] is None, path
    return data


def metric(data, key, digits=2):
    if data is None:
        return '未测'
    if not data['complete_cohort']:
        return f"未完成（{data['successful']}/{data['num_prompts']} 成功）"
    value = data[key]
    return f'{value:.{digits}f}' if value is not None else '—'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=pathlib.Path)
    parser.add_argument('--selection', type=pathlib.Path,
                        help='Explicit relative source paths across run directories')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    selected = {}
    if args.selection:
        root = args.results.resolve()
        for relative in json.loads(args.selection.read_text()):
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Selected source escapes source root: {relative}')
            data = validate(path)
            key = (data['mode'], data['concurrency'])
            if key in selected:
                raise ValueError(f'Multiple selected cohorts for {key}')
            if key[0] not in ['bf16', 'store4'] or key[1] not in [1, 2, 4, 8, 16, 32]:
                raise ValueError(f'Unexpected selected cohort: {key}')
            data['source_path'] = relative
            data['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            selected[key] = data
    rows = []
    for concurrency in [1, 2, 4, 8, 16, 32]:
        arms = {}
        for mode in ['bf16', 'store4']:
            if args.selection:
                arms[mode] = selected.get((mode, concurrency))
            else:
                path = args.results / f'{mode}_in131072_out1024_c{concurrency}.json'
                arms[mode] = validate(path) if path.exists() else None
        bf16, store = arms['bf16'], arms['store4']
        ratio = None
        if bf16 and store:
            assert bf16['payload_sha256'] == store['payload_sha256'], concurrency
            if bf16['complete_cohort'] and store['complete_cohort']:
                ratio = store['output_throughput_tps'] / bf16['output_throughput_tps']
        rows.append({'concurrency': concurrency, 'bf16': bf16, 'store4': store,
                     'store_vs_bf16_output_throughput_ratio': ratio})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    lines = [
        '| 并发 | BF16 输出 tok/s | TQ store 4-bit 输出 tok/s | TQ/BF16 |',
        '| --- | --- | --- | --- |',
    ]
    for row in rows:
        bf16, store = row['bf16'], row['store4']
        ratio = row['store_vs_bf16_output_throughput_ratio']
        values = [str(row['concurrency']), metric(bf16, 'output_throughput_tps'),
                  metric(store, 'output_throughput_tps'), f'{ratio:.3f}x' if ratio else '—']
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '| 并发 | BF16 TTFT均值(s) | TQ TTFT均值(s) | BF16 TPOT均值(ms) | TQ TPOT均值(ms) |',
              '| --- | --- | --- | --- | --- |']
    for row in rows:
        bf16, store = row['bf16'], row['store4']
        values = [str(row['concurrency']), metric(bf16, 'mean_ttft_s'),
                  metric(store, 'mean_ttft_s'), metric(bf16, 'mean_tpot_ms'),
                  metric(store, 'mean_tpot_ms')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.append('\n每格一组同时到达的请求，请求数等于并发；未完成组不计算吞吐或收益比。计时从 HTTP 发送到最后响应结束，排除输入构造，包含排队、prefill、decode。\n')
    args.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
