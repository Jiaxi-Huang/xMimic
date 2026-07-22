# Bundled Motrix wheels

These wheels are built from `../morphos-lab` at commit
`ff1f217cb9f6f17d247642a92a04883897f484a4` on branch
`144-manager-api-archive`:

- `motrix_envs-0.3.0-py3-none-any.whl`
  - SHA-256: `792718b03b950d7b518daf6c8248397f29eb359f9b39453b7cd191c7db0c5533`
- `motrix_rl-0.3.0-py3-none-any.whl`
  - SHA-256: `5e7f55d087f62f728204cf933eed51b90b83ad5866339310a0cb3a6a9701d178`

Rebuild from that checkout with:

```bash
uv build --wheel --out-dir /path/to/xMimic/packages motrix_envs
uv build --wheel --out-dir /path/to/xMimic/packages motrix_rl
```

The repository tracks `packages/*.whl` with Git LFS.
