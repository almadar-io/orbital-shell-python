# Orbital Python Shell

Python host of an Orbital program. The program (the app composed with `std-http-server`) runs on the native kernel through the `orbital_kernel` extension; this project holds the program, the language it runs against, and a small entry point.

## Layout

| Path | What |
|------|------|
| `src/orbital_app/generated/program.orb` | The served program, written by `orb compile` |
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
| `AUTH_PROVIDER` | `firebase` or `oidc`; unset, no token is accepted and every request is anonymous |
| `FIREBASE_PROJECT_ID` | With `firebase`: the project whose ID tokens are accepted |
| `OIDC_ISSUER_URL`, `OIDC_AUDIENCE` | With `oidc`: the issuer and audience tokens must carry |
| `FIREBASE_AUTH_EMULATOR_HOST` | Local dev only: unsigned Auth-emulator tokens are accepted |

The row store is declared by the app (`store:` with a std driver); no variable selects it.

## Sign-in

This project is a server: it has no screen and signs nobody in. A client (the web client compiled with `orb compile --mode client --backend-url <this server>`, a native app, or a script holding a token) obtains the ID token and sends it on every request as `Authorization: Bearer <token>`. The served program verifies the token and binds `@user` to the app's identity row; the host passes the header through as data and verifies nothing. A request with no header is anonymous, and the program signs that session out. Signing out is the client dropping its token.

## Deployment

```bash
./scripts/deploy.sh orbital-app us-central1
./scripts/setup-secrets.sh orbital-app us-central1
```

The Dockerfile copies `src/` and `kernel/` and runs `python -m orbital_app`. The deploy scripts set no sign-in variable: set `AUTH_PROVIDER` and its project or issuer on the service (`gcloud run services update <service> --update-env-vars AUTH_PROVIDER=firebase`), or no token is accepted.
