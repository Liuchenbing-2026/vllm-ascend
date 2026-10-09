#!/usr/bin/env bash
# Run on the build host with GitHub read access; no private key is copied.
set -euo pipefail
task_root=${1:-/data1/tq-128k-ab-20261010}
mkdir -p "$task_root/source-checkouts"

checkout_archive() {
  local name=$1 url=$2 ref=$3 sha=$4 archive=$5
  local checkout="$task_root/source-checkouts/$name"
  if [ -e "$checkout" ]; then
    echo "Existing checkout $checkout; inspect it before reusing" >&2
    return 1
  fi
  git init "$checkout"
  git -C "$checkout" remote add origin "$url"
  git -C "$checkout" fetch --depth 1 origin "$sha"
  git -C "$checkout" checkout --detach FETCH_HEAD
  test "$(git -C "$checkout" rev-parse HEAD)" = "$sha"
  git -C "$checkout" archive --format=tar.gz --output="$task_root/$archive" "$sha"
  echo "Archived $name $ref at $sha"
}

checkout_archive vllm https://github.com/vllm-project/vllm.git v0.28.0 \
  2cf0a6915ce544dc493a0990f2ea38d81601128a vllm-source.tar.gz
checkout_archive vllm-ascend https://github.com/vllm-project/vllm-ascend.git v0.28.0.rc1 \
  96df623103f921df1a4488170d106f36689acb59 ascend-source.tar.gz
checkout_archive tq git@github.com:Liuchenbing-2026/vllm-ascend.git liuchenbing-2026 \
  8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b tq-source.tar.gz

# The task used this exact pinned source from the base image. bootstrap.sh checks
# that it exists and copies it before compiling new native binaries.
echo 'Catlass required: 41bf90da655bba3c66d0acd7e00abe33960ecfd6'
