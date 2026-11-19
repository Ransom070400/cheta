# Releasing cheta

1. Make sure `main` is green: `uv run pytest && uv run ruff check . && uv run mypy cheta`.
2. Set the version in `pyproject.toml` and `cheta/__init__.py`, and add a CHANGELOG entry.
3. Build: `rm -rf dist && uv build` (produces an sdist and a wheel in `dist/`).
4. Test the wheel in a clean environment:
   ```bash
   uv venv /tmp/cheta-check && uv pip install --python /tmp/cheta-check/bin/python dist/*.whl
   /tmp/cheta-check/bin/cheta --help
   /tmp/cheta-check/bin/python -c "import cheta; print(cheta.__version__)"
   ```
5. Try TestPyPI first (needs an account and an API token at https://test.pypi.org):
   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ --token <testpypi-token>
   ```
6. Publish for real (token from https://pypi.org/manage/account/token/):
   ```bash
   uv publish --token <pypi-token>
   ```
   Or set `UV_PUBLISH_TOKEN` instead of passing `--token`. Do not commit tokens.
7. Tag the release: `git tag v0.1.0 && git push origin v0.1.0`.

A PyPI version can never be re-uploaded or replaced, so check the TestPyPI page and the
install test first. If something is wrong, yank the release and publish a new version.
