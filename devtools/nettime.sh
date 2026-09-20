#!/usr/bin/env bash
# Print the current UTC time as "YYYY-MM-DD HH:MM UTC", taken from the HTTP
# `Date` header of a well-known public server rather than the local clock.
# The local clock on the shared GPU box is not reliably NTP-synchronised
# (observed 3.5 minutes slow with `timedatectl` reporting "synchronized: no"),
# so timestamps written into lab notebook entries must come from here.
#
#   ./devtools/nettime.sh          # -> 2026-09-20 03:30 UTC
#   ./devtools/nettime.sh --full   # -> 2026-09-20 03:30:30 UTC
#
# Exits non-zero (and prints nothing on stdout) if no source is reachable;
# the caller must then say the time is unverified instead of guessing.
set -u
fmt='+%Y-%m-%d %H:%M UTC'
[ "${1:-}" = "--full" ] && fmt='+%Y-%m-%d %H:%M:%S UTC'

for url in https://www.google.com https://www.cloudflare.com https://www.microsoft.com; do
    hdr=$(curl -sI --max-time 6 "$url" 2>/dev/null \
        | awk -F': ' 'tolower($1)=="date" {sub(/\r$/, "", $2); print $2}')
    if [ -n "$hdr" ] && t=$(date -u -d "$hdr" "$fmt" 2>/dev/null); then
        echo "$t"
        exit 0
    fi
done

echo "nettime.sh: no network time source reachable" >&2
exit 1
