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

scannet_asset_url() {
  local scene="$1"
  local file_type="$2"

  case "$file_type" in
    .sens)
      printf '%s/%s/%s.sens\n' "${SCANNET_V1_SCANS_URL%/}" "$scene" "$scene"
      ;;
    _vh_clean_2.ply)
      printf '%s/%s/%s_vh_clean_2.ply\n' \
        "${SCANNET_V2_SCANS_URL%/}" "$scene" "$scene"
      ;;
    *)
      printf 'Unsupported ScanNet asset type: %s\n' "$file_type" >&2
      return 1
      ;;
  esac
}

download_asset() {
  local scene="$1"
  local file_type="$2"
  local expected="$3"
  local expected_dir
  local expected_name
  local partial
  local partial_size
  local url
  local attempt
  local -a curl_command
  local -a curl_extra_args

  expected_dir="$(dirname "$expected")"
  expected_name="$(basename "$expected")"
  partial="$expected.partial"

  mkdir -p "$expected_dir"
  cleanup_scannet_asset_staging "$expected"

  if [[ -f "$expected" && -s "$expected" ]]; then
    rm -f -- "$partial"
    printf '[scannet] reuse %s\n' "$expected"
    return 0
  fi
  if [[ -e "$expected" || -L "$expected" ]]; then
    rm -f -- "$expected"
  fi

  url="$(scannet_asset_url "$scene" "$file_type")"
  curl_command=("$SCANNET_CURL")
  if [[ -n "$SCANNET_CURL_ARGS" ]]; then
    read -r -a curl_extra_args <<< "$SCANNET_CURL_ARGS"
    curl_command+=("${curl_extra_args[@]}")
  fi
  command -v "${curl_command[0]}" >/dev/null 2>&1 || {
    printf 'curl command is unavailable: %s\n' "${curl_command[0]}" >&2
    return 1
  }

  for ((attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++)); do
    partial_size=0
    if [[ -f "$partial" ]]; then
      partial_size="$(wc -c < "$partial")"
    fi
    printf '[scannet] download %s (attempt %s/%s, resume bytes: %s)\n' \
      "$url" "$attempt" "$DOWNLOAD_RETRIES" "$partial_size"
    if "${curl_command[@]}" -fL -C - --connect-timeout 30 \
      -o "$partial" "$url"; then
      if [[ -s "$partial" ]] && mv -f -- "$partial" "$expected"; then
        printf '[scannet] ready %s\n' "$expected"
        return 0
      fi
    fi

    printf '[scannet] attempt %s/%s failed for %s\n' \
      "$attempt" "$DOWNLOAD_RETRIES" "$expected" >&2
  done

  printf 'Official ScanNet download did not complete after %s attempts: %s\n' \
    "$DOWNLOAD_RETRIES" "$expected" >&2
  if [[ -s "$partial" ]]; then
    printf 'Partial file retained for the next run: %s (%s bytes)\n' \
      "$partial" "$(wc -c < "$partial")" >&2
  fi
  return 1
}
