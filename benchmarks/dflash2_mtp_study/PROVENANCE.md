# Provenance

Every script here was executed on kylin10-6018 through a file queue that renames the submitted
script to `<id>.done` (or `.failed`) after it runs. **That archived file is the literal byte sequence
that executed**, so the table below is checked against it rather than against recollection.

`sha256` is of the executed bytes. "local file" is the working-copy file whose hash matches exactly.

| task | status | sha256 (16) | file |
|---|---|---|---|
| `20260911t` | ok | `55265aedf63e634a` | audits/drafter_cost_audit_m18_20260911t.sh |
| `20260911u` | ok | `cec78ca9f8869b61` | audits/drafter_cost_audit2_m18_20260911u.sh |
| `20260911v` | ok | `33d964c8fd18ff05` | audits/cardcheck_and_gsm8kshort_m18_20260911v.sh |
| `20260911w` | ok | `3fc8e0bbe5f0c391` | k7 start (sgl template, first attempt) |
| `20260911x` | ok | `c0bf2661032819e1` | k7 bench (sgl template, first attempt) |
| `20260911y` | ok | `fad034835f317997` | audits/mtp_depth_feasibility_m18_20260911y.sh |
| `20260911z` | ok | `42721df9b8231118` | nospec start — **see divergence note** |
| `20260911aa` | ok | `9fd1749d5799f05c` | nospec bench — **see divergence note** |
| `20260911ab` | ok | `793527e07725b393` | k7 start (order-control protocol) |
| `20260911ac` | ok | `f95791819c98639b` | k7 bench (order-control protocol) |
| `20260911ad` | ok | `b716fc389f5db39c` | mtp3 start |
| `20260911ae` | ok | `c9a201740b50c032` | mtp3 bench |
| `20260911af` | FAILED | `8b85faf85b2a97c9` | mtp7 start — killed at the 10-min wall clock while still capturing graphs; the detached container survived and `ag` attached to it |
| `20260911ag` | ok | `7a5a208b89ded33e` | mtp7 bench |
| `20260911ah` | ok | `8d1a421655796c89` | k3 start |
| `20260911ai` | ok | `b379aef8524a3d15` | k3 bench (adds the EOS cells) |
| `20260911aj` | ok | `cdab05897562f525` | k7 start (EOS re-run) |
| `20260911ak` | ok | `3cac2fb833baf07e` | k7 bench (EOS) — order control **failed** (0.9449) |
| `20260911al` | ok | `18f148979b7ee1fa` | mtp3 start (EOS denominators) |
| `20260911am` | ok | `754d948706d2e357` | mtp3 bench (EOS) |
| `20260911an` | ok | `46d872fa99e0cc1c` | nospec start (EOS denominators) |
| `20260911ao` | ok | `a8d9207d080c4b4d` | nospec bench (EOS) |
| `20260911ap` | ok | `bc579d8f585a8b15` | audits/align_gaps_audit_m18_20260911ap.sh |
| `20260912a` | ok | `14df0eef3c36162e` | k7 start (long output) |
| `20260912b` | ok | `b0528fd0631cbb53` | k7 bench (long output) — order control **failed** (0.9233) |
| `20260912c` | ok | `3a8ea0a710b8968f` | mtp3 start (long output) |
| `20260912d` | ok | `acb30d7b16108c2b` | mtp3 bench (long output) |
| `20260912e` | ok | `652d44d034521f41` | audits/provenance_audit_m18_20260912e.sh |
| `20260912f` | ok | `335da5e9c4c72dff` | audits/weights_identity_m18_20260912f.sh |
| `20260912g` | FAILED | `61092558ea309d03` | audits/gsm8k_prompt_fidelity_m18_20260912g.sh — died at exit 141; sections 1–2 completed and are the ones quoted |
| `20260912h` | ok | `bdd33df9c17b5898` | mtp7 start (EOS) |
| `20260912i` | ok | `64b62e552882d32d` | mtp7 bench (EOS) |
| `20260912j` | ok | `8b26ef33eb69175c` | audits/branch_identity_m18_20260912j.sh |

## Divergence note: `20260911z` / `20260911aa`

Of 43 executed scripts, 41 are byte-identical to their working copy. These two are not.

Both are the `nospec` pair, and both were regenerated from the template *after* they had already run,
when the `mtp7` arm was added. The differences are:

- `aa`: one added `mtp7)` branch in a `case "$arm_id"` block.
- `z`: the same `mtp7)` branch, plus a post-readiness step that records the engine's resolved
  `SpeculativeConfig`, whose own `case` acts only for `mtp7` / `mtp3`.

**Both are unreachable when `arm_id=nospec`**, and the recording step runs after the service is up
and readiness has been polled. The `nospec` measurements are unaffected. The per-arm scripts are not
committed for exactly this reason — they are `sed` substitutions of the templates, and keeping 40
near-identical copies in sync with a template that kept changing is what produced this divergence.

## Runtime and weights

- image `vllm-ascend:pr14171-v026-runtime-20260821` = `sha256:e6519803e088d655590cfe3b2ef9b429c1c4509e790f85b1fae4a37acd94ba75`
- runtime manifest `929fb5220edfa11e7804872a14d610b07504ede7b32869a192120bbbc3f71622`
- runtime python tree (1201 files) `d631cbc61f8b751b2f35d0c2a60ab1f40c8486c3fd1e020125e5a4d5e8d585f7`
- vllm python tree (3841 files) `5585d5d4c50e1610c54189ccceda1bb61723bcc2d82a4168db3aebd86d1d59ea`
- draft weights `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` — byte-identical to
  `z-lab/Qwen3.8-27B-DFlash2` (3,848,817,896 bytes, revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`)
- 0 `.py` files in either tree modified during the measurement window (since 2026-09-10)

**The start scripts gate on the manifest's own hash, and that gate is weaker than it looks**: the
manifest is a key=value provenance file, not a `sha256sum` listing, so it does not pin the contents of
the files it describes. An edit to any runtime source file would pass it. The tree hashes above were
computed afterwards and are what a real content gate should check against.
