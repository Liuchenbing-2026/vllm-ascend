# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    csrc = root / "csrc"
    csrc.mkdir(parents=True)
    (root / ".gitmodules").write_text("", encoding="utf-8")
    (csrc / "third_party" / "catlass" / "include").mkdir(parents=True)
    (root / "vllm_ascend" / "_cann_ops_custom").mkdir(parents=True)
    (root / "vllm_ascend" / "_cann_ops_custom" / "old.marker").write_text("keep", encoding="utf-8")

    source = Path(__file__).resolve().parents[2] / "csrc" / "build_aclnn.sh"
    shutil.copy2(source, csrc / "build_aclnn.sh")

    (csrc / "build.sh").write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\n' \"$@\" > build_invocation.txt
printf '%s\\n' \"${CCACHE_DISABLE:-unset}\" > build_ccache_disable.txt
if [[ -e build/preexisting.marker ]]; then
  echo present > build_marker_state.txt
else
  echo absent > build_marker_state.txt
fi
mkdir -p build
cat > build/cann-ops-transformer-fake.run <<'INSTALLER'
#!/bin/bash
set -euo pipefail
if [[ ${FAKE_INSTALL_FAIL:-0} == 1 ]]; then
  exit 9
fi
install_path=${1#--install-path=}
mkdir -p \"$install_path/vendors/custom_transformer/scripts\"
mkdir -p \"$install_path/vendors/custom_transformer/op_api/lib\"
case \"$(uname -m)\" in
  aarch64|arm64) package_arch=aarch64 ;;
  x86_64|amd64) package_arch=x86_64 ;;
  *) exit 81 ;;
esac
if [[ ${FAKE_WRONG_ARCH:-0} == 1 ]]; then
  [[ $package_arch == aarch64 ]] && package_arch=x86_64 || package_arch=aarch64
fi
tiling_lib=\"$install_path/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/$package_arch\"
mkdir -p \"$tiling_lib\"
printf library > \"$install_path/vendors/custom_transformer/op_api/lib/libcust_opapi.so\"
printf opmaster > \"$tiling_lib/libcust_opmaster_rt2.0.so\"
if [[ ${FAKE_INSTALL_INCOMPLETE:-0} == 1 ]]; then
  exit 0
fi
kernel_base=\"$install_path/vendors/custom_transformer/op_impl/ai_core/tbe/kernel\"
kernel_root=\"$kernel_base/ascend910b/sparse_attn_sharedkv\"
config_root=\"$kernel_base/config/ascend910b\"
mkdir -p \"$kernel_root\" \"$config_root\"
printf object > \"$kernel_root/SparseAttnSharedkv_fake.o\"
printf '{}\\n' > \"$kernel_root/SparseAttnSharedkv_fake.json\"
printf '%s\\n' \
  '{\"binList\":[{\"binInfo\":{\"jsonFilePath\":\"ascend910b/sparse_attn_sharedkv/SparseAttnSharedkv_fake.json\"}}]}' \
  > \"$config_root/sparse_attn_sharedkv.json\"
printf '%s\\n' \
  '{\"SparseAttnSharedkv\":{\"binaryList\":[\"SparseAttnSharedkv_fake\"]}}' \
  > \"$config_root/binary_info_config.json\"
if [[ ${FAKE_INVALID_CONFIG:-0} == 1 ]]; then
  printf '{}\\n' > \"$config_root/sparse_attn_sharedkv.json\"
fi
INSTALLER
chmod +x build/cann-ops-transformer-fake.run
""",
        encoding="utf-8",
    )
    os.chmod(csrc / "build.sh", 0o755)
    return root


def run_build(
    fake_repo: Path,
    *,
    reuse: str | None = None,
    ccache: str | None = None,
    install_fail: bool = False,
    install_incomplete: bool = False,
    invalid_config: bool = False,
    wrong_arch: bool = False,
    swap_fail: bool = False,
) -> subprocess.CompletedProcess[str]:
    csrc = fake_repo / "csrc"
    (csrc / "build").mkdir(exist_ok=True)
    (csrc / "build" / "preexisting.marker").write_text("stale", encoding="utf-8")
    (csrc / "output").mkdir(exist_ok=True)
    (csrc / "build_out").mkdir(exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(fake_repo.parent / "home")
    env["GIT_CONFIG_GLOBAL"] = str(fake_repo.parent / "gitconfig")
    env["TMPDIR"] = str(fake_repo.parent / "tmp")
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    for name in (
        "VLLM_ASCEND_ACLNN_REUSE_BUILD",
        "VLLM_ASCEND_ACLNN_CCACHE",
        "FAKE_INSTALL_FAIL",
        "FAKE_INSTALL_INCOMPLETE",
        "FAKE_INVALID_CONFIG",
        "FAKE_WRONG_ARCH",
        "FAKE_SWAP_FAIL",
        "REAL_MV",
    ):
        env.pop(name, None)
    if reuse is not None:
        env["VLLM_ASCEND_ACLNN_REUSE_BUILD"] = reuse
    if ccache is not None:
        env["VLLM_ASCEND_ACLNN_CCACHE"] = ccache
    if install_fail:
        env["FAKE_INSTALL_FAIL"] = "1"
    if install_incomplete:
        env["FAKE_INSTALL_INCOMPLETE"] = "1"
    if invalid_config:
        env["FAKE_INVALID_CONFIG"] = "1"
    if wrong_arch:
        env["FAKE_WRONG_ARCH"] = "1"
    if swap_fail:
        fake_bin = fake_repo / "fake-bin"
        fake_bin.mkdir()
        real_mv = shutil.which("mv")
        assert real_mv is not None
        (fake_bin / "mv").write_text(
            """#!/bin/bash
set -euo pipefail
if [[ ${FAKE_SWAP_FAIL:-0} == 1 ]]; then
  for arg in "$@"; do
    if [[ $arg == *.staging.* ]]; then
      exit 17
    fi
  done
fi
exec "$REAL_MV" "$@"
""",
            encoding="utf-8",
        )
        os.chmod(fake_bin / "mv", 0o755)
        env["FAKE_SWAP_FAIL"] = "1"
        env["REAL_MV"] = real_mv
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(csrc / "build_aclnn.sh"), str(fake_repo), "ascend910b"],
        cwd=csrc,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_build_aclnn_cleans_stale_build_by_default(fake_repo: Path) -> None:
    result = run_build(fake_repo)

    assert result.returncode == 0, result.stderr
    assert (fake_repo / "csrc" / "build_marker_state.txt").read_text(encoding="utf-8").strip() == "absent"
    invocation = (fake_repo / "csrc" / "build_invocation.txt").read_text(encoding="utf-8").splitlines()
    assert invocation[-2:] == ["--ccache", "false"]
    assert (fake_repo / "csrc" / "build_ccache_disable.txt").read_text(encoding="utf-8").strip() == "1"
    installed = fake_repo / "vllm_ascend" / "_cann_ops_custom"
    assert (installed / ".gitkeep").is_file()
    assert not (installed / "old.marker").exists()


def test_build_aclnn_reuses_build_only_when_explicit(fake_repo: Path) -> None:
    result = run_build(fake_repo, reuse="1")

    assert result.returncode == 0, result.stderr
    assert (fake_repo / "csrc" / "build_marker_state.txt").read_text(encoding="utf-8").strip() == "present"


def test_build_aclnn_rejects_invalid_reuse_value(fake_repo: Path) -> None:
    result = run_build(fake_repo, reuse="yes")

    assert result.returncode == 2
    assert "invalid VLLM_ASCEND_ACLNN_REUSE_BUILD" in result.stderr
    assert not (fake_repo / "csrc" / "build_invocation.txt").exists()
    assert (fake_repo / "csrc" / "build" / "preexisting.marker").is_file()


def test_build_aclnn_rejects_invalid_ccache_before_cleanup(fake_repo: Path) -> None:
    result = run_build(fake_repo, ccache="maybe")

    assert result.returncode == 2
    assert "invalid VLLM_ASCEND_ACLNN_CCACHE" in result.stderr
    assert not (fake_repo / "csrc" / "build_invocation.txt").exists()
    assert (fake_repo / "csrc" / "build" / "preexisting.marker").is_file()


def test_build_aclnn_keeps_installed_package_when_staged_install_fails(fake_repo: Path) -> None:
    result = run_build(fake_repo, install_fail=True)

    assert result.returncode == 9
    installed = fake_repo / "vllm_ascend" / "_cann_ops_custom"
    assert (installed / "old.marker").read_text(encoding="utf-8") == "keep"
    assert not list((fake_repo / "vllm_ascend").glob("_cann_ops_custom.staging.*"))


def test_build_aclnn_rejects_incomplete_kernel_package(fake_repo: Path) -> None:
    result = run_build(fake_repo, install_incomplete=True)

    assert result.returncode == 1
    assert "missing non-empty kernel objects or metadata" in result.stderr
    installed = fake_repo / "vllm_ascend" / "_cann_ops_custom"
    assert (installed / "old.marker").read_text(encoding="utf-8") == "keep"
    assert not list((fake_repo / "vllm_ascend").glob("_cann_ops_custom.staging.*"))


def test_build_aclnn_rejects_invalid_kernel_config(fake_repo: Path) -> None:
    result = run_build(fake_repo, invalid_config=True)

    assert result.returncode == 1
    assert "has no binList entries" in result.stderr
    installed = fake_repo / "vllm_ascend" / "_cann_ops_custom"
    assert (installed / "old.marker").read_text(encoding="utf-8") == "keep"


def test_build_aclnn_rejects_opmaster_for_wrong_host_architecture(fake_repo: Path) -> None:
    result = run_build(fake_repo, wrong_arch=True)

    assert result.returncode == 1
    assert "missing libcust_opmaster_rt2.0.so" in result.stderr
    assert (fake_repo / "vllm_ascend" / "_cann_ops_custom" / "old.marker").is_file()


def test_build_aclnn_lock_covers_cleanup_and_build(fake_repo: Path) -> None:
    fcntl = pytest.importorskip("fcntl")
    install_dir = (fake_repo / "vllm_ascend" / "_cann_ops_custom").resolve()
    cksum = subprocess.run(
        ["cksum"],
        input=str(install_dir),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()[0]
    lock_path = fake_repo.parent / "tmp" / f"vllm_ascend_build_aclnn.{cksum}.lock"
    lock_path.parent.mkdir(exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_build(fake_repo)

    assert result.returncode == 1
    assert "another custom-op build or package installation is in progress" in result.stderr
    assert not (fake_repo / "csrc" / "build_invocation.txt").exists()
    assert (fake_repo / "csrc" / "build" / "preexisting.marker").is_file()


def test_build_aclnn_restores_installed_package_when_swap_fails(fake_repo: Path) -> None:
    result = run_build(fake_repo, swap_fail=True)

    assert result.returncode == 17
    installed = fake_repo / "vllm_ascend" / "_cann_ops_custom"
    assert (installed / "old.marker").read_text(encoding="utf-8") == "keep"
    assert not list((fake_repo / "vllm_ascend").glob("_cann_ops_custom.staging.*"))
    assert not list((fake_repo / "vllm_ascend").glob("_cann_ops_custom.backup.*"))
