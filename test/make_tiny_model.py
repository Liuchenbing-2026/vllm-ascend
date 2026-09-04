"""Generate a tiny GLM-5.3-Flash checkpoint with the EXACT tensor naming/layout of the
real ZhipuAI release (unpacked per-expert MoE, split q/k/v_conv1d, real MTP layer).

Derives every tensor name+shape+dtype from the real checkpoint's safetensors headers,
then writes random data. Keeps:
  layers 0-7 verbatim  (0,1,2 = dense+KDA ; 3 = DSA+MoE ; 4,5,6 = KDA+MoE ; 7 = DSA+MoE)
  real layer 45 (MTP) renumbered to 8
  experts 0..N_EXPERTS-1 only
  vision blocks 0..VIS_DEPTH-1 only
  full vocab embed/lm_head (tokenizer must stay valid)
"""
import json, os, re, struct, sys, shutil
import numpy as np

SRC = os.environ.get("GLM53_SRC", "/data02/GLM-5.3-Flash-BF16")
DST = sys.argv[1] if len(sys.argv) > 1 else "/data02/glm53_tiny/model2"
N_LAYERS  = int(os.environ.get("N_LAYERS", 8))
N_EXPERTS = int(os.environ.get("N_EXPERTS", 16))
VIS_DEPTH = int(os.environ.get("VIS_DEPTH", 4))
REAL_MTP_LAYER = 45
REAL_N_EXPERTS = 288
MAX_SHARD = 4 * 1024**3

DT = {"BF16": np.dtype(np.uint16), "F16": np.dtype(np.float16), "F32": np.dtype(np.float32),
      "F64": np.dtype(np.float64), "I64": np.dtype(np.int64), "I32": np.dtype(np.int32),
      "I8": np.dtype(np.int8), "U8": np.dtype(np.uint8), "BOOL": np.dtype(np.bool_)}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def write_safetensors(path, tensors):
    """tensors: list of (name, dtype_str, shape). Random data."""
    header, off = {}, 0
    for name, dts, shape in tensors:
        nb = int(np.prod(shape)) * DT[dts].itemsize if shape else DT[dts].itemsize
        header[name] = {"dtype": dts, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    rng = np.random.default_rng(0)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, dts, shape in tensors:
            n = int(np.prod(shape)) if shape else 1
            if dts == "BF16":
                # small normal values, encoded as the top 16 bits of fp32
                a = (rng.standard_normal(n).astype(np.float32) * 0.02).view(np.uint32)
                f.write((a >> 16).astype(np.uint16).tobytes())
            elif dts in ("F16", "F32", "F64"):
                f.write((rng.standard_normal(n) * 0.02).astype(DT[dts]).tobytes())
            elif dts == "BOOL":
                f.write(np.zeros(n, DT[dts]).tobytes())
            else:
                f.write(np.zeros(n, DT[dts]).tobytes())
    return os.path.getsize(path)


def main():
    idx = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    meta = {}
    for s in shards:
        h = read_header(os.path.join(SRC, s))
        for k, v in h.items():
            if k != "__metadata__":
                meta[k] = (v["dtype"], tuple(v["shape"]))

    out = []          # (new_name, dtype, shape)
    for name, (dts, shape) in meta.items():
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", name)
        if m:
            li, rest = int(m.group(1)), m.group(2)
            if li == REAL_MTP_LAYER:
                new_li = N_LAYERS                      # MTP layer sits right after the stack
            elif li < N_LAYERS:
                new_li = li
            else:
                continue
            e = re.match(r"mlp\.experts\.(\d+)\.", rest)
            if e and int(e.group(1)) >= N_EXPERTS:
                continue
            # the router gate carries the expert count in dim 0 -> shrink it too
            if rest.startswith("mlp.gate.") and shape and shape[0] == REAL_N_EXPERTS:
                shape = (N_EXPERTS,) + tuple(shape[1:])
            out.append(("model.language_model.layers.%d.%s" % (new_li, rest), dts, shape))
            continue
        b = re.match(r"model\.visual\.blocks\.(\d+)\.", name)
        if b and int(b.group(1)) >= VIS_DEPTH:
            continue
        out.append((name, dts, shape))

    out.sort()
    total = sum(int(np.prod(s)) * DT[d].itemsize for _, d, s in out)
    print("tensors: %d   total: %.2f GiB" % (len(out), total / 1024**3))

    os.makedirs(DST, exist_ok=True)
    groups, cur, cur_sz = [], [], 0
    for t in out:
        sz = int(np.prod(t[2])) * DT[t[1]].itemsize
        if cur and cur_sz + sz > MAX_SHARD:
            groups.append(cur); cur, cur_sz = [], 0
        cur.append(t); cur_sz += sz
    if cur:
        groups.append(cur)

    weight_map = {}
    for i, g in enumerate(groups, 1):
        fn = "model-%05d-of-%05d.safetensors" % (i, len(groups))
        b = write_safetensors(os.path.join(DST, fn), g)
        for name, _, _ in g:
            weight_map[name] = fn
        print("  wrote %s (%d tensors, %.2f GiB)" % (fn, len(g), b / 1024**3))

    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(DST + "/model.safetensors.index.json", "w"), indent=2)

    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "processor_config.json", "configuration.json"):
        p = os.path.join("/data02/glm53_tiny/src", f)
        if os.path.exists(p):
            shutil.copy2(p, DST)
    print("done ->", DST)


main()
