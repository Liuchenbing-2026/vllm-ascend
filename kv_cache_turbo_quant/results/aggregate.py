import json, glob, os
d = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*_in*_c*.json"))):
    name = os.path.basename(f)[:-5]
    label, rest = name.split("_in", 1)
    inp, conc = rest.split("_c")
    j = json.load(open(f))
    rows[(label, int(inp), int(conc))] = j

def g(j, k):
    return j.get(k)

keys = sorted(set((i,c) for (l,i,c) in rows))
print(f"{'inp':>6} {'conc':>5} | {'base_ttft':>9} {'shad_ttft':>9} {'d_ttft%':>8} | {'base_tpot':>9} {'shad_tpot':>9} {'d_tpot%':>8} | {'base_tps':>9} {'shad_tps':>9} {'d_tps%':>8}")
for (inp, conc) in keys:
    b = rows.get(("baseline", inp, conc)); s = rows.get(("shadow", inp, conc))
    if not b or not s:
        print(inp, conc, "MISSING"); continue
    bt, st = b["mean_ttft_ms"], s["mean_ttft_ms"]
    bp, sp = b["mean_tpot_ms"], s["mean_tpot_ms"]
    bo, so = b["output_throughput"], s["output_throughput"]
    bt2, st2 = b["total_token_throughput"], s["total_token_throughput"]
    print(f"{inp:>6} {conc:>5} | {bt:>9.1f} {st:>9.1f} {(st/bt-1)*100:>7.1f}% | {bp:>9.2f} {sp:>9.2f} {(sp/bp-1)*100:>7.1f}% | {bo:>9.1f} {so:>9.1f} {(so/bo-1)*100:>7.1f}%")
    print(f"{''':>6} {''':>5} | {'total tok/s':>9} {bt2:>9.1f} {st2:>9.1f} {(st2/bt2-1)*100:>7.1f}%")
