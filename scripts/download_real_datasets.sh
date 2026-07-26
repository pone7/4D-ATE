#!/bin/bash
# Attempt to download real AET datasets from multiple known sources.
# Strategy: try multiple mirrors for each dataset, log results per target,
# do not abort on failure.
set -u
cd /data/aet-bench
mkdir -p data/real_raw
cd data/real_raw

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a download.log; }
try() {
  local name="$1"; shift
  local dest="$1"; shift
  local url="$1"; shift
  log "→ $name: trying $url"
  if curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 900 \
          -o "$dest.part" "$url" 2>&1 | tail -5 >> download.log; then
    if [ -s "$dest.part" ]; then
      mv "$dest.part" "$dest"
      log "✅ $name: got $(stat -c%s "$dest") bytes"
      return 0
    fi
  fi
  rm -f "$dest.part"
  log "❌ $name: failed"
  return 1
}

log "================ start downloads ================"

# ========================================================================
# 1. Jiang et al. 2016 Scientific Data — multi-sample AET dataset on Dryad
#    DOI: 10.5061/dryad.m20q3
#    Samples: Pt/C, PtCu, Co2P, W-needle
# ========================================================================
mkdir -p jiang2016_scidata
(
  cd jiang2016_scidata
  # Dryad gives per-file download URLs. Landing page:
  #   https://datadryad.org/stash/dataset/doi:10.5061/dryad.m20q3
  # Each file has /api/v2/files/<id>/download
  # We can't guess file IDs, so fetch the dataset listing first.
  try "dryad-listing" "dataset.json" \
    "https://datadryad.org/api/v2/datasets/doi:10.5061%2Fdryad.m20q3"
  if [ -f dataset.json ]; then
    # Extract file URLs
    /opt/conda/envs/aet4dgs/bin/python -c "
import json, sys
with open('dataset.json') as f:
    d = json.load(f)
for link in d.get('_links', {}).get('stash:files', []):
    print(link.get('href', ''))
" 2>&1 | tee files_index.txt || true
  fi
) &
DRYAD_PID=$!

# ========================================================================
# 2. Yuan et al. 2022 MEA/HEA (Nature 611, 2022)
#    Miao group, UCLA — data DOI should be on Zenodo or UCLA Dash
# ========================================================================
mkdir -p yuan2022_hea
(
  cd yuan2022_hea
  # Try Zenodo search
  try "zenodo-hea-search" "search.json" \
    "https://zenodo.org/api/records?q=title%3A%22atomic+electron+tomography%22+high+entropy&size=20"
) &
ZENODO_HEA_PID=$!

# ========================================================================
# 3. Yang et al. 2017 Nature FePt (UCLA)
#    Originally at physics.ucla.edu; try Zenodo + Nature supplementary
# ========================================================================
mkdir -p yang2017_fept
(
  cd yang2017_fept
  try "zenodo-fept-search" "search.json" \
    "https://zenodo.org/api/records?q=title%3A%22FePt%22+atomic+tomography&size=20"
  try "zenodo-fept-search2" "search2.json" \
    "https://zenodo.org/api/records?q=%22Yang%22+%22FePt%22+electron+tomography&size=20"
) &
ZENODO_FEPT_PID=$!

# ========================================================================
# 4. Miao group AET GitHub repositories that might bundle data
# ========================================================================
mkdir -p miao_github
(
  cd miao_github
  # Search GitHub for Miao AET repos
  try "github-miao-search" "search.json" \
    "https://api.github.com/search/repositories?q=atomic+electron+tomography+user:miaolab&per_page=20"
  try "github-aet-search" "search_aet.json" \
    "https://api.github.com/search/repositories?q=atomic+electron+tomography&per_page=30"
) &
GH_PID=$!

# ========================================================================
# 5. Nature Scientific Data figshare repositories
# ========================================================================
mkdir -p figshare_aet
(
  cd figshare_aet
  try "figshare-search" "search.json" \
    "https://api.figshare.com/v2/articles/search?search_for=atomic+electron+tomography&item_type=3"
) &
FS_PID=$!

# ========================================================================
# 6. Ophus group (materials AET, pyMultislicer, abTEM, etc.)
# ========================================================================
mkdir -p ophus_github
(
  cd ophus_github
  try "ophus-repos" "search.json" \
    "https://api.github.com/search/repositories?q=user:cophus+tomography&per_page=20"
) &
OPHUS_PID=$!

wait $DRYAD_PID $ZENODO_HEA_PID $ZENODO_FEPT_PID $GH_PID $FS_PID $OPHUS_PID
log "================ initial searches done ================"
