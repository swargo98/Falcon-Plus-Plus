#!/bin/bash
set -euo pipefail

output_dir="/mnt/nvme0n1/src"
file_count=10
file_size=1200      # MiB
parallel_jobs=10

mkdir -p "$output_dir"

create_file() {
  local file_index=$1
  local path="$output_dir/random_file_$file_index"
  echo "Generating $path (${file_size} MiB)..."
  # AES-CTR pseudorandom stream, exact size via fullblock
  openssl enc -aes-256-ctr -nosalt -pass pass:"seed-$file_index" </dev/zero 2>/dev/null \
    | dd of="$path" bs=1M count="$file_size" iflag=fullblock status=none conv=fdatasync
  # sanity: show exact size in bytes
  stat -c '%n %s bytes' "$path"
}

export -f create_file
export output_dir file_size

seq 1 "$file_count" | xargs -n1 -P "$parallel_jobs" -I{} bash -c 'create_file "$@"' _ {}
echo "Created $file_count files of ${file_size} MiB in $output_dir"