"""Earth Engine authentication and initialisation.

Rule: always pass an explicit project ID. Default projects from Cloud and Colab
do not have Earth Engine enabled, and relying on them is the most common cause
of the "project is not registered" error.
"""

from __future__ import annotations

import logging

import ee

from .config import require_env

logger = logging.getLogger(__name__)

_INITIALISED = False

_HELP = """
Earth Engine could not initialise. Work through these in order:

  1. Is GEE_PROJECT_ID in .env the project ID (e.g. gdp-proxy-472913), not the name?
  2. Is the Earth Engine API enabled on THAT project?
     https://console.cloud.google.com/apis/library/earthengine.googleapis.com
  3. Is that project registered for noncommercial use?
     https://code.earthengine.google.com/register
  4. Have you run `earthengine authenticate`? If credentials are stale:
     `earthengine authenticate --force`

Full error reference is in SETUP.md.
"""


def _service_account_credentials():
    """Earth Engine credentials from a service-account key, or None.

    CI has no ``earthengine authenticate`` credentials, so the scheduled check
    would fail without this. ``google-github-actions/auth`` writes the key to a
    file and points ``GOOGLE_APPLICATION_CREDENTIALS`` at it; that is the path
    documented by gee-community/ee-initialize-github-actions.

    Returns None when the variable is unset, so local runs keep using the
    interactive credentials rather than being forced onto a service account.
    """
    import json
    import os
    from pathlib import Path

    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not key_path:
        return None
    path = Path(key_path)
    if not path.exists():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS points at {path}, which does not exist."
        )
    email = json.loads(path.read_text(encoding="utf-8")).get("client_email")
    if not email:
        raise RuntimeError(f"{path} has no client_email; is it a service-account key?")
    logger.info("Using Earth Engine service account %s", email)
    return ee.ServiceAccountCredentials(email, str(path))


def init_ee(project_id: str | None = None, force: bool = False) -> str:
    """Initialise the Earth Engine client. Returns the project ID used."""
    global _INITIALISED
    project = project_id or require_env("GEE_PROJECT_ID")

    if _INITIALISED and not force:
        return project

    try:
        credentials = _service_account_credentials()
        if credentials is not None:
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001 - we re-raise with guidance
        raise RuntimeError(f"{exc}\n{_HELP}") from exc

    _INITIALISED = True
    logger.info("Earth Engine initialised on project %s", project)
    return project
