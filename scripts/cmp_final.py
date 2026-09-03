import json, re, statistics as st

def load(p):
    d = json.load(open(p))
    return {r["id"]: r for r in d["requests"] if r and r.get("ok")}, d

def gsm_ans(t):
    m = re.findall(r"####\s*(-?\d[\d,.]*)", t)
    if m: return m[-1].replace(",", "")
    for line in reversed(t.splitlines()):
        m = re.findall(r"-?\d[\d,.]*", line)
        if m: return m[-1].replace(",", "")
    return None

def mmlu_ans(t):
    m = re.search(r"Answer:\s*([A-D])", t)
    if m: return m.group(1)
    m = re.findall(r"\b([A-D])\b", t[-200:])
    return m[-1] if m else None

def med(xs): return round(st.median(xs), 1) if xs else None

def cmp(name, base_f, py_f, judge):
    b, bd = load(base_f); p, pd = load(py_f)
    ids = sorted(set(b) & set(p))
    print(f"=== {name}: n={len(ids)} (base {bd['n']}/{sum(1 for x in bd['requests'] if x)}ok, py {pd['n']}/{sum(1 for x in pd['requests'] if x)}ok)")
    ident = sum(1 for i in ids if b[i]["text"] == p[i]["text"])
    print(f"  text-identity: {ident}/{len(ids)} ({100.0*ident/len(ids):.1f}%)")
    tb = [b[i]["ttft_ms"] for i in ids if b[i].get("ttft_ms") is not None]
    tp = [p[i]["ttft_ms"] for i in ids if p[i].get("ttft_ms") is not None]
    eb = [b[i]["e2e_ms"] for i in ids if b[i].get("e2e_ms") is not None]
    ep = [p[i]["e2e_ms"] for i in ids if p[i].get("e2e_ms") is not None]
    mb, mp = med(tb), med(tp); meb, mep = med(eb), med(ep)
    if mb and mp: print(f"  ttft: base {mb}ms vs py {mp}ms = {100.0*(mp-mb)/mb:+.1f}%")
    if meb and mep: print(f"  e2e:  base {meb}ms vs py {mep}ms = {100.0*(mep-meb)/meb:+.1f}%")
    if judge:
        cb = sum(1 for i in ids if judge(b[i]["text"]) == judge(b[i]["ref"]))
        cp = sum(1 for i in ids if judge(p[i]["text"]) == judge(p[i]["ref"]))
        bj = [judge(b[i]["text"]) == judge(b[i]["ref"]) for i in ids]
        pj = [judge(p[i]["text"]) == judge(p[i]["ref"]) for i in ids]
        fl = sum(1 for x, y in zip(bj, pj) if not x and y)
        fr = sum(1 for x, y in zip(bj, pj) if x and not y)
        print(f"  acc: base {100.0*cb/len(ids):.1f}% vs py {100.0*cp/len(ids):.1f}% = {100.0*(cp-cb)/len(ids):+.1f}pt (fixed {fl} / broke {fr})")
        ai = sum(1 for i in ids if judge(b[i]["text"]) == judge(p[i]["text"]))
        print(f"  answer-identity: {100.0*ai/len(ids):.1f}%")

cmp("gsm8k  full", "/tmp/evals/gsm8k_ascendc.json", "/tmp/evals/gsm8k_py_full.json", gsm_ans)
cmp("mmlu   full", "/tmp/evals/mmlu_ascendc.json", "/tmp/evals/mmlu_py.json", mmlu_ans)
cmp("humaneval (no-score)", "/tmp/evals/humaneval_ascendc.json", "/tmp/evals/humaneval_py.json", None)
