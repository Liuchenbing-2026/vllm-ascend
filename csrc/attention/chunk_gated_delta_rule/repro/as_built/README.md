# The kernel as built, byte for byte

The five headers in `ops-transformer-9.1.0/` are the exact bytes that were
compiled into the custom opp vendor package (`gdrcust_transformer`) used for
every measurement in `../REPRODUCE.md`. They are here so that "the code that ran"
is a file you can checksum, not a claim.

## Provenance chain

Upstream base: `gitcode.com/cann/ops-transformer`, branch `9.1.0`, commit
`0684247214bae5abb5cb6dc7d90b34a6ad401cdd`. The two patches in `patches/` apply
on top of it and produce exactly these files.

Every link verified by md5 on 2026-09-13:

| File | source tree | packaged into the `.run` | installed under `$ASCEND_OPP_PATH/vendors/gdrcust_transformer/` |
|---|---|---|---|
| `chunk_gated_delta_rule.h` | `61d304ed` | `61d304ed` | `61d304ed` |
| `chunk_gated_delta_rule_matmul_basic.h` | `806d1f38` | `806d1f38` | `806d1f38` |
| `chunk_gated_delta_rule_stage1.h` | `ffddbaee` | `ffddbaee` | `ffddbaee` |
| `chunk_gated_delta_rule_stage2.h` | `ad38f55c` | `ad38f55c` | `ad38f55c` |
| `chunk_gated_delta_rule_stage3.h` | `8b3fa725` | `8b3fa725` | `8b3fa725` |

Full md5s are in `MANIFEST.md5`.

The compiled artifact that this produced, still installed on the measurement
machine, is
`op_impl/ai_core/tbe/kernel/ascend910b/chunk_gated_delta_rule/ChunkGatedDeltaRule_b8eaf39689b840ee88390d15b17183a4.o`,
md5 `1d90ce859e2fc3e5929d638f85502966` -- which is the id quoted in the second
patch's subject line.

Three files ship in the same package and are **not** modified by this work; they
are listed only so a diff of the installed vendor does not look unaccounted for:
`chunk_gated_delta_rule_utils.h` `9e1bd428`,
`chunk_gated_delta_rule_tiling_data.h` `4e6788dc`,
`chunk_gated_delta_rule_proto.h` `79880ea7`.

## Two copies of the same kernel, in two layouts

`../../op_kernel/arch22/*.h` in this branch is the same kernel in **vllm-ascend's**
layout, where the headers sit one level down in `arch22/` next to `arch35/`, so
the tiling header is included as `"../chunk_gated_delta_rule_tiling_data.h"`.

`ops-transformer-9.1.0/*.h` here is the same kernel in **ops-transformer 9.1.0's**
layout, which is flat, so the same include has no `../`. That is the one
difference, and `../REPRODUCE.md` section 4 already documents the `sed` that
converts between them.

Applying that `sed` to the `arch22/` copy reproduces four of these five files
byte for byte. The fifth, `stage2.h`, differs by **exactly two comment lines**
(`// Dv_ 必须能被 f 整除...` and `// 同上: Dv_ 为奇数时禁用 Dv 分片`), which were
written when the code was published for review and are therefore absent from the
copy that was compiled. No code line differs, so the `.o` is unaffected -- but it
is the reason the md5s of that one file do not match, and guessing at that
instead of recording it is how provenance rots.

## Rebuilding it

```bash
git clone -b 9.1.0 https://gitcode.com/cann/ops-transformer.git
cd ops-transformer
git checkout 0684247214bae5abb5cb6dc7d90b34a6ad401cdd
git am /path/to/this/dir/patches/*.patch
# or, without git: cp ops-transformer-9.1.0/*.h attention/chunk_gated_delta_rule/op_kernel/

bash build.sh --pkg --soc=ascend910b --vendor_name=gdrcust --ops=chunk_gated_delta_rule -j64
echo "build exit=$?"        # a failed build leaves the previous .run in place
./cann-ops-transformer-gdrcust_linux-aarch64.run --quiet
```

Then verify the routing two ways before trusting any number -- `../REPRODUCE.md`
section 5 has both, and neither is optional: the maps check alone only proves the
host-side `libcust_opapi.so` was loaded.

## Why the patches are not upstream

`gitcode.com/cann/ops-transformer` is Huawei's repository and this work has no
write access to it, so the branch these commits were made on
(`arch22-cgdr-opt`) exists only on the measurement machine. This directory is
the publication path: the patches, the resulting bytes, and the checksums that
tie them to the binary that produced the numbers.
