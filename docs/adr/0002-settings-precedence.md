# ADR-0002 — Settings precedence (env > TOML > defaults)

Status: Accepted.

## Context

Three sources need to coexist: hardcoded defaults, an optional
`settings.toml`, and `GAZECONTROL_*` environment variables. We use
`pydantic-settings` for validation.

## Decision

`AppSettings.settings_customise_sources` returns the canonical order:

```python
(init_settings, env_settings, TomlConfigSettingsSource(...))
```

This makes precedence: **constructor kwargs > env > TOML > defaults**.
A module-level singleton (`get_settings()`) caches the resolved instance
behind a double-checked lock, so concurrent first-callers do not race.

## Consequences

- Test fixtures can override anything via `AppSettings(...)` without
  touching env or files.
- Operators can flip behaviour at deploy time via env vars without
  shipping a TOML.
- The singleton is intentionally immutable for the process lifetime;
  hot-reload is out of scope (see roadmap).
