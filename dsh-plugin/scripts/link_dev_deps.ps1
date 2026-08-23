# Development-mode dependency link (Windows).
#
# When developing against a pnpm-installed deepseek-harness SOURCE checkout
# (instead of the published rc packages), link this project's node_modules to
# the checkout's pnpm virtual store so the plugin resolves the EXACT same
# package versions the checkout runtime uses — both for `tsc` and at runtime.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/link_dev_deps.ps1 `
#       -RepoRoot C:\path\to\deepseek-harness
#   # or: $env:DSH_REPO_ROOT = 'C:\path\to\deepseek-harness' first
#
# Remove the link with `Remove-Item node_modules` (junction only) before
# running `pnpm install` from the registry.
param(
    [string]$RepoRoot = $env:DSH_REPO_ROOT
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    throw 'Pass -RepoRoot <deepseek-harness checkout> or set $env:DSH_REPO_ROOT first.'
}
$repo = (Resolve-Path $RepoRoot).Path
$store = Join-Path $repo 'node_modules\.pnpm\node_modules'
if (-not (Test-Path (Join-Path $store '@deepseek-ai'))) {
    throw "pnpm virtual store not found under $repo - run `pnpm install` in the checkout first."
}

$plugin = Split-Path -Parent $PSScriptRoot
$storeDir = Join-Path $plugin '.dev-store'
$link = Join-Path $plugin 'node_modules'

if (Test-Path $link) {
    $item = Get-Item $link -Force
    if ($item.LinkType -eq 'Junction' -and $item.Target -eq $storeDir) {
        Write-Host "dev link already present: $link -> $storeDir"
        exit 0
    }
    throw "$link exists and is not the dev junction; remove it manually first."
}

# A junction cannot point INTO another junction path on some setups, so build
# a small store of real junctions: @deepseek-ai (runtime + types), @types and
# typescript + .bin (so `pnpm build` finds tsc), then junction node_modules to
# that store.
New-Item -ItemType Directory -Path $storeDir -Force | Out-Null
New-Item -ItemType Junction -Path (Join-Path $storeDir '@deepseek-ai') -Target (Join-Path $store '@deepseek-ai') | Out-Null
$types = Join-Path $repo 'node_modules\@types'
if (Test-Path $types) {
    New-Item -ItemType Junction -Path (Join-Path $storeDir '@types') -Target $types | Out-Null
}
$ts = Join-Path $repo 'node_modules\typescript'
if (Test-Path $ts) {
    New-Item -ItemType Junction -Path (Join-Path $storeDir 'typescript') -Target $ts | Out-Null
}
$bin = Join-Path $repo 'node_modules\.bin'
if (Test-Path $bin) {
    New-Item -ItemType Junction -Path (Join-Path $storeDir '.bin') -Target $bin | Out-Null
}
New-Item -ItemType Junction -Path $link -Target $storeDir | Out-Null
Write-Host "linked $link -> $storeDir"
Write-Host "plugin now resolves @deepseek-ai/* from the checkout ($repo)"
