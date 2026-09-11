# LiteLLM 1.91.0 PR #34462 backport

This is a minimal backport of the OpenAI-compatible chat path from
BerriAI/litellm squash commit `bc8d2b65399b00569482e4ad15671b40952c9a52`.

Included:

- Anthropic single-image `tool_result` becomes a structured `image_url` part.
- Structured tool-result images are hoisted after each consecutive tool-message
  run, with upstream's placeholder and tool-output boundary text.
- `ChatCompletionToolMessage.content` admits image parts.
- Sync/async regression coverage for single, mixed, multiple, parallel, and
  text-only tool results.

Not included because this deployment uses the OpenAI-compatible chat path:

- Responses API adapter changes
- Azure-specific transform hook
- Gemini lint-only change
- UI schema and unrelated upstream changes

The original container files and the pre-backport Compose file are preserved in
`backups/1.91.0-20260911T101823/`. The complete upstream squash patch is stored
at `upstream/bc8d2b6539.patch`.

## Regression test

Run the test container with the four source mounts from `docker-compose.yml`,
mount `tests/` at `/backport-tests`, and execute:

```sh
/app/.venv/bin/python3 /backport-tests/run_regression.py
```

## Rollback

From `/opt/litellm`:

```sh
cp backports/pr34462/backups/1.91.0-20260911T101823/docker-compose.yml docker-compose.yml
docker compose up -d --no-deps --force-recreate litellm
```

The installed files inside the image were never overwritten; removing the four
bind mounts restores the exact original LiteLLM 1.91.0 code.
