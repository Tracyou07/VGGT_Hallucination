#!/usr/bin/env bash

cleanup_scannet_asset_staging() {
  local expected="$1"
  local expected_dir
  local expected_name
  local staging_prefix
  local staging_root

  expected_dir="$(dirname "$expected")"
  expected_name="$(basename "$expected")"
  staging_prefix="$expected_dir/.scannet-download-$expected_name.staging."

  (
    shopt -s nullglob
    for staging_root in "$staging_prefix"*; do
      rm -rf -- "$staging_root"
    done
  )
}

download_asset() {
  local scene="$1"
  local file_type="$2"
  local expected="$3"
  local expected_dir
  local expected_name
  local staging_prefix
  local staging_root
  local staged
  local attempt
  local attempt_succeeded

  expected_dir="$(dirname "$expected")"
  expected_name="$(basename "$expected")"
  staging_prefix="$expected_dir/.scannet-download-$expected_name.staging."

  mkdir -p "$expected_dir"
  cleanup_scannet_asset_staging "$expected"

  if [[ -f "$expected" && -s "$expected" ]]; then
    printf '[scannet] reuse %s\n' "$expected"
    return 0
  fi
  if [[ -e "$expected" || -L "$expected" ]]; then
    rm -f -- "$expected"
  fi

  for ((attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++)); do
    staging_root="$(mktemp -d "${staging_prefix}${attempt}.XXXXXX")"
    staged="$staging_root/scans/$scene/$expected_name"
    attempt_succeeded=0

    if printf '\n\n\n\n' | python "$SCANNET_DOWNLOAD_SCRIPT" \
      -o "$staging_root" --id "$scene" --type "$file_type"; then
      if [[ -f "$staged" && -s "$staged" ]] \
        && mv -f -- "$staged" "$expected"; then
        attempt_succeeded=1
      fi
    fi

    if ! rm -rf -- "$staging_root"; then
      printf 'Failed to remove ScanNet staging root: %s\n' "$staging_root" >&2
      return 1
    fi
    if [[ "$attempt_succeeded" == "1" ]]; then
      return 0
    fi

    printf '[scannet] attempt %s/%s failed for %s\n' \
      "$attempt" "$DOWNLOAD_RETRIES" "$expected" >&2
  done

  printf 'Official downloader did not produce %s after %s attempts.\n' \
    "$expected" "$DOWNLOAD_RETRIES" >&2
  return 1
}
