# Changelog fragments

Each unreleased change adds its own small file here instead of editing
`CHANGELOG.md` directly. This way two branches never touch the same lines and
never conflict. At release time, `towncrier build` collapses these fragments
into `CHANGELOG.md` and removes them.

## Naming

`<issue-or-id>.<type>.md` where `<type>` is one of:

- `added`
- `changed`
- `fixed`
- `removed`
- `security`

Example: `42.feat-probe-command.added.md`

## Content

Write the entry as finished prose. It is copied verbatim into the changelog.

```
Add a `--probe` mode for quick endpoint reachability checks.
```
