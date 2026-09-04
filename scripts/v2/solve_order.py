"""Enumerate accumulation orders against harvested AscendC tie samples."""
import json
import struct


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def add(a, b):
    return f32(a + b)


def mul(a, b):
    return f32(a * b)


def to_bf16_bits(x):
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    lower = b & 0xFFFF
    upper = b >> 16
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper = (upper + 1) & 0xFFFF
    return upper


def seq(ps):
    a = ps[0]
    for p in ps[1:]:
        a = add(a, p)
    return a


def adj_tree(ps):
    while len(ps) > 1:
        ps = [add(ps[i], ps[i + 1]) for i in range(0, len(ps), 2)]
    return ps[0]


def fold_tree(ps):
    while len(ps) > 1:
        h = len(ps) // 2
        ps = [add(ps[i], ps[i + h]) for i in range(h)]
    return ps[0]


def zero_seq(ps):          # 0 + p0 + p1 ... (leading zero changes nothing)
    a = 0.0
    for p in ps:
        a = add(a, p)
    return a


ORDERS = {
    "seq16": seq,
    "rev_seq16": lambda ps: seq(ps[::-1]),
    "adj_tree16": adj_tree,
    "fold_tree16": fold_tree,
    "blk8adj_pair": lambda ps: add(adj_tree(ps[:8]), adj_tree(ps[8:])),
    "blk8fold_pair": lambda ps: add(fold_tree(ps[:8]), fold_tree(ps[8:])),
    "blk8seq_pair": lambda ps: add(seq(ps[:8]), seq(ps[8:])),
    "pairs_seq8": lambda ps: seq([add(ps[2 * k], ps[2 * k + 1]) for k in range(8)]),
    "pairs_fold8": lambda ps: fold_tree([add(ps[2 * k], ps[2 * k + 1]) for k in range(8)]),
    "quad_seq": lambda ps: seq([seq(ps[4 * k:4 * k + 4]) for k in range(4)]),
    "quad_adj": lambda ps: adj_tree([adj_tree(ps[4 * k:4 * k + 4]) for k in range(4)]),
    "even_odd_seq": lambda ps: add(seq(ps[0::2]), seq(ps[1::2])),
    "even_odd_adj": lambda ps: add(adj_tree(ps[0::2]), adj_tree(ps[1::2])),
    "blk8adj_pair_rev": lambda ps: add(adj_tree(ps[8:]), adj_tree(ps[:8])),
}

samples = json.load(open("tie_samples.json"))
print("samples:", len(samples))
tallies = {k: 0 for k in ORDERS}
for s in samples:
    ps = [mul(x, w) for x, w in zip(s["x"], s["w"])]
    refb = s["ref_bits"] & 0xFFFF
    for name, fn in ORDERS.items():
        acc = fn(ps)
        out = to_bf16_bits(add(acc, f32(s["y0"])))
        if out == refb:
            tallies[name] += 1
for name, n in sorted(tallies.items(), key=lambda kv: -kv[1]):
    print("%-18s %d/%d" % (name, n, len(samples)))
