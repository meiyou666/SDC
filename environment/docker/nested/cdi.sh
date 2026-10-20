#!/usr/bin/env bash
set -euo pipefail

# nvidia-ctk may append .yaml: always provide the extension, outside CDI search paths.
config=${SDC_NESTED_CONFIG:-/etc/sdc-docker/settings.conf}
if [[ -f "$config" ]]; then source "$config"; fi
cdi_dir=${SDC_NESTED_CDI_DIR:-/etc/cdi}
mkdir -p "$cdi_dir"
stage=$(mktemp -d "$(dirname "$cdi_dir")/.sdc-nvidia.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ $(stat -c %d "$stage") != $(stat -c %d "$cdi_dir") ]]; then
    echo 'CDI staging and destination must share a filesystem for atomic replacement.' >&2
    exit 1
fi
# Quarantine only this bootstrap's remnants, keeping them available for rollback.
shopt -s nullglob
backup=
for remnant in "$cdi_dir"/.sdc-nvidia.*; do
    if [[ -f "$remnant" || -L "$remnant" ]]; then
        if [[ -z "$backup" ]]; then
            backup_root=${SDC_NESTED_ETC:-/etc/sdc-docker}/cdi-backups
            mkdir -p "$backup_root"
            backup=$(mktemp -d "$backup_root/remnants.XXXXXXXX")
        fi
        mv -- "$remnant" "$backup/"
    fi
done
nvidia-ctk cdi generate --output "$stage/nvidia.yaml"
[[ -s "$stage/nvidia.yaml" ]] || { echo 'nvidia-ctk produced an empty CDI spec.' >&2; exit 1; }
chmod 0644 "$stage/nvidia.yaml"
mv -f -- "$stage/nvidia.yaml" "$cdi_dir/nvidia.yaml"
