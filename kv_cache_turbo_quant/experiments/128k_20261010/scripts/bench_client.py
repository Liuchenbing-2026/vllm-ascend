#!/usr/bin/env python3
"""Measure one simultaneous request cohort with exact token IDs and SSE timings."""
import argparse
import asyncio
import codecs
import hashlib
import json
import pathlib
import random
import statistics
import time
from datetime import datetime, timezone

import aiohttp


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    position = (len(xs) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (position - lo)


async def measure(args):
    count = args.num_prompts or args.concurrency
    payloads = []
    hashes = []
    for request_id in range(count):
        generator = random.Random(args.seed + request_id)
        tokens = generator.choices(range(1000, 10000), k=args.input_len)
        payload = {
            'model': args.model, 'prompt': tokens, 'max_tokens': args.output_len,
            'temperature': 0.0, 'seed': args.seed, 'ignore_eos': True,
            'stream': True, 'stream_options': {'include_usage': True},
        }
        raw = json.dumps(payload, separators=(',', ':')).encode()
        payloads.append(raw)
        hashes.append(hashlib.sha256(raw).hexdigest())
    records = [{'id': i, 'status': 'pending', 'events': 0, 'text': ''} for i in range(count)]
    progress_path = pathlib.Path(args.output).with_suffix('.progress.json')
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async def request(session, index):
        record = records[index]
        async with semaphore:
            beginning = time.perf_counter()
            record.update(status='running', started_s=beginning - started)
            first = None
            last = None
            carry = ''
            decoder = codecs.getincrementaldecoder('utf-8')()
            try:
                async with session.post(args.url, data=payloads[index],
                                        headers={'Content-Type': 'application/json'}) as response:
                    record['http_status'] = response.status
                    if response.status != 200:
                        raise RuntimeError((await response.text())[:4000])
                    async for raw in response.content.iter_any():
                        carry += decoder.decode(raw)
                        while '\n\n' in carry:
                            event, carry = carry.split('\n\n', 1)
                            for line in event.splitlines():
                                if not line.startswith('data: '):
                                    continue
                                value = line[6:]
                                if value == '[DONE]':
                                    continue
                                message = json.loads(value)
                                if 'error' in message:
                                    raise RuntimeError(str(message['error']))
                                if message.get('usage'):
                                    record['usage'] = message['usage']
                                for choice in message.get('choices', []):
                                    fragment = choice.get('text', '')
                                    if fragment:
                                        now = time.perf_counter()
                                        first = first or now
                                        last = now
                                        record['events'] += 1
                                        record['text'] += fragment
                                        record['last_token_s'] = now - beginning
                                    if choice.get('finish_reason') is not None:
                                        record['finish_reason'] = choice['finish_reason']
                usage = record.get('usage', {})
                if usage.get('prompt_tokens') != args.input_len:
                    raise RuntimeError(f'Unexpected input work: {usage}')
                if usage.get('completion_tokens') != args.output_len:
                    raise RuntimeError(f'Unexpected output work: {usage}')
                if first is None or last is None:
                    raise RuntimeError('No generated text received')
                record.update(status='success', ttft_s=first - beginning,
                              tpot_s=(last - first) / max(args.output_len - 1, 1))
            except Exception as error:
                record.update(status='failed', error_type=type(error).__name__, error=str(error))
                if first is not None:
                    record['partial_ttft_s'] = first - beginning
            finally:
                record['latency_s'] = time.perf_counter() - beginning

    async def monitor():
        while True:
            progress_path.write_text(json.dumps({
                'elapsed_s': time.perf_counter() - started,
                'requests': [{k: v for k, v in r.items() if k != 'text'} for r in records],
            }, indent=2) + '\n')
            await asyncio.sleep(5)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        monitoring = asyncio.create_task(monitor())
        await asyncio.gather(*(request(session, i) for i in range(count)))
        monitoring.cancel()
        try:
            await monitoring
        except asyncio.CancelledError:
            pass
    duration = time.perf_counter() - started
    successful = [r for r in records if r['status'] == 'success']
    all_success = len(successful) == count
    ttfts = [r['ttft_s'] for r in successful]
    tpots = [r['tpot_s'] for r in successful]
    latencies = [r['latency_s'] for r in successful]
    result = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'concurrency': args.concurrency, 'num_prompts': count,
        'input_len': args.input_len, 'output_len': args.output_len,
        'seed': args.seed, 'timeout_s': args.timeout, 'payload_sha256': hashes,
        'successful': len(successful), 'failed': count - len(successful),
        'duration_s': duration, 'complete_cohort': all_success,
        'output_throughput_tps': count * args.output_len / duration if all_success else None,
        'total_throughput_tps': count * (args.input_len + args.output_len) / duration if all_success else None,
        'mean_ttft_s': statistics.mean(ttfts) if ttfts else None,
        'p99_ttft_s': percentile(ttfts, 0.99),
        'mean_tpot_ms': statistics.mean(tpots) * 1000 if tpots else None,
        'p99_tpot_ms': percentile(tpots, 0.99) * 1000 if tpots else None,
        'mean_latency_s': statistics.mean(latencies) if latencies else None,
        'requests': records,
        'timing_boundary': 'cohort HTTP send to last SSE/usage response; payload generation excluded',
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'payload_sha256')}, ensure_ascii=False))
    return 0 if all_success else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18377/v1/completions')
    parser.add_argument('--model', default='qwen3-30b-tq-ab')
    parser.add_argument('--mode', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--num-prompts', type=int)
    parser.add_argument('--input-len', type=int, default=131072)
    parser.add_argument('--output-len', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(measure(args)))


if __name__ == '__main__':
    main()
