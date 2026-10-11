#!/usr/bin/env python3
"""Fresh public-weight fingerprints on the new host, before model trials."""
from pathlib import Path
import hashlib,json,concurrent.futures,datetime

def digest(path):
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return {'file':path.name,'bytes':after.st_size,'sha256':h.hexdigest(),'mtime_ns':after.st_mtime_ns}

r=Path('/ws/artifacts');m=Path('/models/Qwen3-30B-A3B')
old=json.loads((r/'model-weights-sha256.json').read_text());initial=json.loads((r/'model-manifest.json').read_text());started=datetime.datetime.now(datetime.timezone.utc).isoformat()
config={k:digest(m/k)for k in initial['files']}
assert all(config[k]['sha256']==v['sha256']for k,v in initial['files'].items())
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:weights=list(pool.map(digest,sorted(m.glob('*.safetensors'))))
assert {v['file']:(v['bytes'],v['sha256'])for v in weights}=={v['file']:(v['bytes'],v['sha256'])for v in old['weight_files']}
out={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(m),'config':config,'weight_files':weights,'all_config_and_weight_bytes_match_original_full_sha256':True,'origin_revision':'unverified; hashes match prior public-weight snapshot'}
(r/'model-third-sha256.json').write_text(json.dumps(out,indent=2)+'\n');print('MODEL_HASHES_MATCH',len(weights),flush=True)
