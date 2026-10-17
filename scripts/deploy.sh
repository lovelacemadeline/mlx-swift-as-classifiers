#!/usr/bin/env bash
# deploy.sh — Mac -> iPhone for the ClassifierProbe app (story 6.1), adapted from
# iphone-exploration's scripts/deploy.sh (same signing, install, launch, log flow).
#
#   ./scripts/deploy.sh               build, install, launch (full: text + vision), stream logs
#   ./scripts/deploy.sh --check       environment preflight only, no build
#   ./scripts/deploy.sh --build       build and install only
#   ./scripts/deploy.sh --push-model  copy the E2B snapshot into the app's Documents/ (after a first install)
#   ./scripts/deploy.sh --measure     build, install, push the model, run full then --text-only
#   ./scripts/deploy.sh --run [ARGS]  launch the installed app (ARGS: --text-only | --kernels), wait
#                                     for DONE in its log, copy the log back to .build/ClassifierProbe/
#   ./scripts/deploy.sh --logs        stream logs from an already-running app
#   ./scripts/deploy.sh --pull        copy Documents/probe-log.txt back
#
# Signing: free Apple ID Personal Team, profiles expire after 7 days, so every
# build re-signs. The team id is NOT committed: scripts/deploy.env (gitignored)
# holds DEVELOPMENT_TEAM=XXXXXXXXXX, or pass it in the environment.
# The model is never bundled in the .app (4 GiB): --push-model copies it into the
# app data container, where it survives reinstalls of the same bundle id.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$REPO_ROOT/App/ClassifierProbe/ClassifierProbe.xcodeproj"
TARGET="ClassifierProbe"
BUNDLE_ID="com.maddie.classifierprobe"
CONFIG="${CONFIG:-Release}"
BUILD_DIR="$REPO_ROOT/.build/ClassifierProbe"
APP="$BUILD_DIR/$TARGET.app"
DEVICE_ID="${DEVICE_ID:-6DC9DA18-3FF0-5538-84C3-218744FCEDD6}"
LOG_SUBSYSTEM="com.maddie.classifierprobe"   # Logger subsystem in ProbeRun.swift
MODEL_REPO="mlx-community--gemma-4-E2B-it-qat-4bit"
MODEL_REV="${MODEL_REV:-42f62737af7a9fd8c1d55d79666c1a217be4e2e2}"
MODEL_DIR_NAME="gemma-4-E2B-it-qat-4bit"      # modelDirName in ProbeRun.swift
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"

[[ -f "$REPO_ROOT/scripts/deploy.env" ]] && . "$REPO_ROOT/scripts/deploy.env"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m/!\\\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxxx\033[0m %s\n' "$*" >&2; exit 1; }

preflight() {
  command -v xcodebuild >/dev/null || die "xcodebuild not found (install Xcode)."
  xcrun -f metal >/dev/null 2>&1 || die "no Metal toolchain: xcodebuild -downloadComponent MetalToolchain"
  xcrun devicectl list devices 2>/dev/null | grep "$DEVICE_ID" | grep -qE "available|connected" \
    || die "device $DEVICE_ID not available to devicectl — plugged in and unlocked?"
  security find-identity -v -p codesigning 2>/dev/null | grep -q 'Apple Development' \
    || die "no Apple Development identity: sign in to Xcode > Settings > Accounts (one-time, manual)."
  if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then
    DEVELOPMENT_TEAM="$(defaults read com.apple.dt.Xcode IDEProvisioningTeamByIdentifier 2>/dev/null \
      | awk -F'[ ;]+' '/teamID/ {print $3; exit}')"
    [[ -n "$DEVELOPMENT_TEAM" ]] || die "DEVELOPMENT_TEAM unset and not in Xcode prefs; put it in scripts/deploy.env"
    export DEVELOPMENT_TEAM
    warn "recovered DEVELOPMENT_TEAM from Xcode prefs; persist it in scripts/deploy.env"
  fi
  local snap; snap="$(snapshot_dir)"
  [[ -f "$snap/model.safetensors" ]] || warn "model snapshot $MODEL_REV not in the HF cache (needed for --push-model)"
  say "preflight ok — device $DEVICE_ID available, team set, $CONFIG build"
}

snapshot_dir() {
  local hub="${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}"
  hub="${hub:-$HOME/.cache/huggingface/hub}"
  echo "$hub/models--$MODEL_REPO/snapshots/$MODEL_REV"
}

build() {
  say "building $TARGET ($CONFIG, re-signing; 7-day Personal Team profile)"
  mkdir -p "$BUILD_DIR"
  # Device destination + -allowProvisioningDeviceRegistration: see the exploration
  # repo's deploy.sh for why (a fresh team cannot mint a profile otherwise).
  # -skip*Validation: swift-syntax macros in the package graph need a trust prompt otherwise.
  xcodebuild \
    -project "$PROJECT" -scheme "$TARGET" \
    -destination "platform=iOS,id=$DEVICE_ID" \
    -configuration "$CONFIG" \
    -derivedDataPath "$BUILD_DIR/DerivedData" \
    -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
    -skipMacroValidation -skipPackagePluginValidation \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" \
    DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" \
    build
  [[ -d "$APP" ]] || die "build reported success but $APP is missing."
  say "built $APP"
}

install() {
  say "installing to device"
  xcrun devicectl device install app --device "$DEVICE_ID" "$APP"
}

push_model() {
  local snap; snap="$(snapshot_dir)"
  [[ -f "$snap/model.safetensors" ]] || die "no snapshot at $snap"
  # The HF cache holds symlinks into blobs/; stage real files (APFS clones, no extra disk).
  local stage="$BUILD_DIR/model-stage/$MODEL_DIR_NAME"
  rm -rf "$stage"; mkdir -p "$stage"
  for f in config.json generation_config.json chat_template.jinja processor_config.json \
           tokenizer.json tokenizer_config.json model.safetensors model.safetensors.index.json; do
    cp -c -L "$snap/$f" "$stage/$f" 2>/dev/null || cp -L "$snap/$f" "$stage/$f"
  done
  say "copying $(du -shL "$stage" | cut -f1) to $BUNDLE_ID:Documents/$MODEL_DIR_NAME (devicectl skips unchanged files)"
  xcrun devicectl device copy to --device "$DEVICE_ID" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
    --source "$stage" --destination "Documents/$MODEL_DIR_NAME" --timeout 3600
}

launch() {
  say "launching $BUNDLE_ID $*"
  xcrun devicectl device process launch --device "$DEVICE_ID" --terminate-existing "$BUNDLE_ID" "$@"
}

pull() {
  mkdir -p "$BUILD_DIR"
  xcrun devicectl device copy from --device "$DEVICE_ID" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
    --source Documents/probe-log.txt --destination "$BUILD_DIR/probe-log.txt" -q
}

run_and_wait() {
  launch "$@"
  local t=0 out="$BUILD_DIR/probe-log.txt"
  rm -f "$out"
  while (( t < RUN_TIMEOUT )); do
    sleep 10; t=$((t + 10))
    pull 2>/dev/null || continue
    if grep -qE ' (DONE|ERROR |MODEL_MISSING)' "$out"; then break; fi
    # Gone without DONE: killed (jetsam) or crashed.
    if ! xcrun devicectl device info processes --device "$DEVICE_ID" 2>/dev/null | grep -q "$TARGET.app"; then
      warn "app is no longer running and never logged DONE — check jetsam reports (Settings > Privacy > Analytics)"
      break
    fi
  done
  say "log at $out"
  cat "$out"
}

stream_logs() {
  say "streaming logs (Ctrl-C to stop) — CLASSIFIER_PROBE lines"
  # libimobiledevice wants the hardware UDID, not the CoreDevice UUID (exploration story 1.1).
  local udid="${LOG_DEVICE_UDID:-$(idevice_id -l 2>/dev/null | head -1)}"
  [[ -n "$udid" ]] || die "no device visible to libimobiledevice (idevice_id -l)."
  idevicesyslog -u "$udid" 2>/dev/null | grep --line-buffered -E "CLASSIFIER_PROBE|$LOG_SUBSYSTEM|jetsam|$TARGET"
}

case "${1:---all}" in
  --check)      preflight ;;
  --build)      preflight; build; install ;;
  --push-model) push_model ;;
  --run)        shift; run_and_wait "$@" ;;
  --measure)    preflight; build; install; push_model
                run_and_wait; cp "$BUILD_DIR/probe-log.txt" "$BUILD_DIR/probe-log-full.txt"
                run_and_wait --text-only; cp "$BUILD_DIR/probe-log.txt" "$BUILD_DIR/probe-log-text-only.txt" ;;
  --logs)       stream_logs ;;
  --pull)       pull && cat "$BUILD_DIR/probe-log.txt" ;;
  --all|"")     preflight; build; install; launch; stream_logs ;;
  -h|--help)    sed -n '2,20p' "$0" ;;
  *)            die "unknown option: $1 (try --help)" ;;
esac
