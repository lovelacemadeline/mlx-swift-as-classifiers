---
type: runbook
status: active
title: iPhone deploy — ClassifierProbe build, install, model push and run
owner: maddie
updated: 2026-10-17
---

# iPhone deploy — ClassifierProbe build, install, model push and run

`scripts/deploy.sh` takes the `ClassifierProbe` app (`App/ClassifierProbe/`, story 6.1)
from this Mac to the iPhone 17 over the cable. It is adapted from
`iphone-exploration/scripts/deploy.sh` (same Personal Team signing, `devicectl`
install and launch, `idevicesyslog` streaming). Verified 2026-10-17, Xcode 26.6,
iOS 26.6.1 phone, CoreDevice id `6DC9DA18-3FF0-5538-84C3-218744FCEDD6`.

## One-time setup

- `scripts/deploy.env` (gitignored, never committed) holds `DEVELOPMENT_TEAM=<team id>`;
  copy it from `iphone-exploration/scripts/deploy.env`, or let `--check` recover it from Xcode's prefs.
- **Trust the developer once:** the first launch of a newly signed app fails with
  `Unable to launch ... profile has not been explicitly trusted` until someone on the phone taps
  Settings > General > VPN & Device Management > the Apple Development profile > Trust.
- **The free Personal Team allows three installed sideloaded apps per device.**
  Install fails with `This device has reached the maximum number of installed apps
  using a free developer profile` (MIInstallerErrorDomain 13) while three are on the
  phone. Free a slot first, e.g. the WebDriverAgent runner (no user data; rebuilt from
  source by the exploration repo):
  `xcrun devicectl device uninstall app --device <id> com.maddie.iexp.WDARunner.xctrunner`.
  Do not remove UTM / UTM SE without saving their VMs.

## Commands

| command | does |
|---|---|
| `./scripts/deploy.sh --check` | preflight: Xcode, Metal toolchain, device available, signing identity, team, model snapshot |
| `./scripts/deploy.sh --build` | `xcodebuild -destination 'platform=iOS,id=<device>'` (Release; `CONFIG=Debug` overrides), then install |
| `./scripts/deploy.sh --push-model` | stage the E2B snapshot (APFS clones of the HF cache's symlinked blobs) and `devicectl device copy to --domain-type appDataContainer --domain-identifier com.maddie.classifierprobe --destination Documents/gemma-4-E2B-it-qat-4bit` |
| `./scripts/deploy.sh --run [--text-only \| --kernels]` | launch (`--kernels`: story 5.1's kernel probe, no model), poll `Documents/probe-log.txt` back every 10 s until `DONE`/`ERROR`, or until the process is gone (jetsam or crash) |
| `./scripts/deploy.sh --measure` | all of the above, then a full run and a `--text-only` run; logs land in `.build/ClassifierProbe/probe-log-{full,text-only}.txt` |
| `./scripts/deploy.sh --logs` | stream `CLASSIFIER_PROBE` lines (subsystem `com.maddie.classifierprobe`) |

## Facts

- The model is never bundled in the `.app` (38 MB without it). It must be pushed **after**
  a first install (the data container exists only then) and survives reinstalls of the
  same bundle id. The checkpoint is one 4.03 GiB `model.safetensors` (audio, vision and
  text tensors in the same shard, see `model.safetensors.index.json`), so the whole file
  ships; `sanitize` drops the audio tensors after the read.
- MLXLMCommon's `loadWeights` evaluates every tensor in the file before `sanitize`, so
  the load peak includes the 581 MiB audio tower even though it is never kept.
- The Personal Team profile **does** grant `com.apple.developer.kernel.increased-memory-limit`
  (checked in `embedded.mobileprovision`); profiles expire after 7 days, every build re-signs.
- **Cap MLX's buffer cache.** With the default cache the first run was killed by jetsam
  (`vm-pageshortage`, 5.34 GiB resident) on the first fixture: 0.57 GiB of freed buffers sat
  in MLX's cache after load. The probe sets `Memory.cacheLimit` to 256 MiB (`--cache-mib N`)
  and calls `Memory.clearCache()` after load. Jetsam reports: `xcrun devicectl device info
  files --domain-type systemCrashLogs`, then `device copy from` the `JetsamEvent-*.ips`.
- `--text-only` loads with `MLXRunner.load(..., includeVision: false)`: no vision tower is
  built and its weights are dropped like audio's.
- Two device ids: `devicectl` takes the CoreDevice UUID, `idevicesyslog` the hardware UDID
  (`idevice_id -l`); the script asks for each.
