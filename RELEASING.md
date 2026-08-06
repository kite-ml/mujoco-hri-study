# Releasing

`mujoco-hri-study` publishes to PyPI through **Trusted Publishing** (OIDC). GitHub
proves the workflow's identity to PyPI directly, so there is no API token anywhere —
nothing to store in secrets, leak in a log, or rotate. The trade is a one-time setup
on PyPI that has to be done by hand, by someone signed in to the account.

## One-time setup

Do this once, before the first release. Steps 1–3 are on pypi.org and cannot be
scripted.

**1. Have a PyPI account with 2FA enabled.** 2FA is mandatory for publishing.

**2. Add a pending publisher.** The project does not exist on PyPI yet, so use the
*pending* publisher form — it pre-authorises the workflow to create the project on
its first upload.

Go to <https://pypi.org/manage/account/publishing/> and fill in:

| Field | Value |
| --- | --- |
| PyPI project name | `mujoco-hri-study` |
| Owner | `kite-ml` |
| Repository name | `mujoco-hri-study` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The environment name must match `environment: pypi` in
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). If you leave it
blank on PyPI, remove that line from the workflow too, or every publish will be
rejected for a mismatch.

**3. Repeat on TestPyPI** (optional but recommended, so you can rehearse):
<https://test.pypi.org/manage/account/publishing/>, same values.

**4. Create the GitHub environment.** In the repo: Settings → Environments → New
environment → name it `pypi`. Adding yourself as a required reviewer means every
publish waits for a human click, which is worth it for an irreversible action —
**a version number on PyPI can never be reused, even after deletion.**

## Cutting a release

**1. Bump the version in all three places.** They must agree; the workflow refuses
to publish if the tag disagrees with `pyproject.toml`.

- `pyproject.toml` → `version`
- `src/mjhri/__init__.py` → `__version__`
- `CITATION.cff` → `version`

**2. Rehearse on TestPyPI.** Actions → Publish → Run workflow → target `testpypi`.
Then check it installs from a clean environment:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ mujoco-hri-study
```

The extra index is needed because mujoco/numpy/pydantic are not on TestPyPI.

**3. Tag and release.** Publishing a GitHub Release triggers the real upload:

```bash
git tag v0.1.0 && git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --generate-notes
```

**4. Verify.**

```bash
pip install mujoco-hri-study
python -c "import mjhri; print(mjhri.__version__)"
```

**5. Update the README.** The install section says the package is not on PyPI yet;
once it is, that becomes `pip install mujoco-hri-study` and the Python badge can
point back at the PyPI project page.

## What the workflow does

Before anything is uploaded it installs the package, fetches the robot meshes so the
scene-backed tests actually run rather than skipping, runs the suite, checks the tag
against `pyproject.toml`, builds, and runs `twine check`. A red test is a failed
release, not a bad publish.

## If a release goes wrong

You cannot re-upload a version, and deleting one does not free the number. Yank it
(PyPI → Manage → Yank) so it stops being installed by default but existing pins keep
resolving, then publish a patch version. Yanking is the recoverable move; deleting
is not.
