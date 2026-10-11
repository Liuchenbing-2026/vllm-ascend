#!/usr/bin/env python3
"""Save fixed single-request greedy token IDs; excluded from performance."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = []
    for length, output_len in [(20, 20), (2000, 200)]:
        for offset in [0, 1]:
            payload = {'model': 'qwen3-30b-tq-ab',
                       'prompt': random.Random(20261010 + offset).choices(range(1000, 10000), k=length),
                       'max_tokens': output_len, 'temperature': 0.0, 'seed': 20261010,
                       'ignore_eos': True, 'stream': False, 'logprobs': 1,
                       'return_tokens_as_token_ids': True}
            raw = json.dumps(payload, separators=(',', ':')).encode()
            start = time.perf_counter()
            request = urllib.request.Request('http://127.0.0.1:18377/v1/completions',
                      data=raw, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=1200) as response:
                result = json.load(response)
            assert result['usage']['prompt_tokens'] == length
            assert result['usage']['completion_tokens'] == output_len
            choice = result['choices'][0]
            assert choice['finish_reason'] == 'length'
            assert len(choice['logprobs']['tokens']) == output_len
            records.append({'input_len': length, 'output_len': output_len, 'seed_offset': offset,
                            'payload_sha256': hashlib.sha256(raw).hexdigest(),
                            'duration_s': time.perf_counter() - start, 'response': result})
            print(json.dumps({'mode': args.mode, 'input_len': length, 'seed_offset': offset,
                              'complete': True}), flush=True)
            Path(args.output).write_text(json.dumps({'mode': args.mode,
                'scope': 'four fixed single-request greedy samples, not a semantic quality benchmark',
                'records': records}, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
