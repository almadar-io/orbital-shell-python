# Orbital Python Shell

Python host of an Orbital program. The program (the app composed with `std-http-server`) runs on the native kernel through the `orbital_kernel` extension; this project holds the program, the language it runs against, and a small entry point.

## Layout

| Path | What |
|------|------|
| `src/orbital_app/generated/program.json` | The served program, written by `orb compile` |
| `src/orbital_app/generated/language.json` | The language bundle the kernel runs against |
| `src/orbital_app/main.py` | Entry point: loads both, reads the program's locale catalogs from `locales/`, calls `orbital_kernel.serve` |
| `kernel/` | The `orbital_kernel` extension (or set `ORBITAL_KERNEL_DIR`) |
| `locales/<locale>.json` | One catalog per locale the program declares (the app fails to start if one is missing) |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
python -m orbital_app
```

The server listens on port 3030 (`PORT` replaces it).

## Environment

| Variable | Description |
|----------|-------------|
| `PORT` | Server port (default 3030) |
| `ORBITAL_KERNEL_DIR` | Directory holding `orbital_kernel.abi3.so` (default `./kernel`) |
| `DATA_BACKEND` | The row store: `mock`, `firebase`, `postgres`, `couchdb` |
| `DATABASE_URL`, `COUCHDB_URL`, `FIREBASE_*` | Store connection, read by the kernel |

## Deployment

```bash
./scripts/deploy.sh orbital-app us-central1
./scripts/setup-secrets.sh orbital-app us-central1
```

The Dockerfile copies `src/` and `kernel/` and runs `python -m orbital_app`.
