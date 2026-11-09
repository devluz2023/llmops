#!/bin/sh
set -eu

if [ ! -f /data/homeserver.yaml ]; then
    python -m synapse.app.homeserver \
        --server-name=matrix.local \
        --config-path=/data/homeserver.yaml \
        --generate-config \
        --report-stats=no
fi

python - <<'PY'
from pathlib import Path
import secrets
import yaml

config_path = Path("/data/homeserver.yaml")
config = yaml.safe_load(config_path.read_text())
config["registration_shared_secret"] = (
    config.get("registration_shared_secret") or secrets.token_urlsafe(32)
)
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

exec python -m synapse.app.homeserver --config-path=/data/homeserver.yaml
