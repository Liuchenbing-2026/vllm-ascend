import base64, io, json, struct, time, urllib.request, zlib

def make_png(w, h, rgb):
    """Minimal solid-colour PNG, no PIL needed."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))

png = make_png(224, 224, (220, 30, 30))   # solid red
b64 = base64.b64encode(png).decode()
print("png bytes:", len(png))

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": "What single colour fills this image? Answer with one word."},
    ]}],
    "max_tokens": 128, "temperature": 0,
}
req = urllib.request.Request(URL + "/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    m = r["choices"][0]["message"]
    print("OK %.2fs prompt_tokens=%d completion=%d" % (dt, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]))
    print("content:", repr((m.get("content") or "")[-300:]))
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:1500])
except Exception as e:
    print("FAILED:", type(e).__name__, e)
