"""Phase 0 exit test.

Run: python -m gdp_proxy.doctor

Checks each link in the setup chain separately so a failure tells you which
step in SETUP.md to go back to, rather than a generic Earth Engine error.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime

from .config import ConfigError, country_config, require_env

PASS = "PASS"
FAIL = "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def check_env() -> tuple[Check, str | None]:
    try:
        project = require_env("GEE_PROJECT_ID")
    except ConfigError as exc:
        return Check("Environment", FAIL, str(exc), "SETUP.md Part A step 2"), None
    if " " in project or project != project.lower():
        return (
            Check(
                "Environment",
                FAIL,
                f"GEE_PROJECT_ID '{project}' looks like a project NAME",
                "Use the project ID from the Cloud console, e.g. gdp-proxy-472913",
            ),
            None,
        )
    return Check("Environment", PASS, f"GEE_PROJECT_ID={project}"), project


def check_import() -> Check:
    try:
        import ee  # noqa: F401
    except ImportError as exc:
        return Check("earthengine-api installed", FAIL, str(exc), "pip install -e '.[dev]'")
    return Check("earthengine-api installed", PASS, "import ee succeeded")


def check_credentials() -> Check:
    try:
        import ee

        ee.data.get_persistent_credentials()
    except Exception as exc:  # noqa: BLE001
        return Check(
            "Local credentials",
            FAIL,
            str(exc)[:200],
            "Run: earthengine authenticate --force",
        )
    return Check("Local credentials", PASS, "credentials file found")


def check_initialise(project: str) -> Check:
    from .auth import init_ee

    try:
        init_ee(project)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "not registered" in msg or "not signed up" in msg.lower():
            remedy = "Register the project at https://code.earthengine.google.com/register"
        elif "has not been used" in msg or "disabled" in msg:
            remedy = (
                "Enable the API at "
                "https://console.cloud.google.com/apis/library/earthengine.googleapis.com "
                "then wait 3 minutes"
            )
        elif "permission" in msg.lower():
            remedy = "Authenticate with the account that owns the project"
        else:
            remedy = "See the error table in SETUP.md"
        return Check("Earth Engine init", FAIL, msg.split("\n")[0][:200], remedy)
    return Check("Earth Engine init", PASS, f"initialised on {project}")


def check_compute() -> Check:
    import ee

    try:
        value = ee.Number(1).add(1).getInfo()
    except Exception as exc:  # noqa: BLE001
        return Check("Round trip compute", FAIL, str(exc)[:200], "See SETUP.md error table")
    if value != 2:
        return Check("Round trip compute", FAIL, f"expected 2, got {value}", "")
    return Check("Round trip compute", PASS, "1 + 1 = 2 on the server")


def check_viirs(cfg: dict) -> Check:
    import ee

    dataset = cfg["viirs_monthly"]
    try:
        coll = ee.ImageCollection(dataset)
        size = coll.size().getInfo()
        latest_ms = coll.aggregate_max("system:time_start").getInfo()
    except Exception as exc:  # noqa: BLE001
        return Check(
            f"VIIRS {dataset}", FAIL, str(exc)[:200], "Check the dataset ID in the catalog"
        )
    latest = datetime.utcfromtimestamp(latest_ms / 1000).strftime("%Y-%m")
    return Check(f"VIIRS {dataset}", PASS, f"{size} images, newest composite {latest}")


def check_boundaries_reachable() -> Check:
    import requests

    try:
        resp = requests.head("https://geodata.ucdavis.edu/gadm/gadm4.1/", timeout=15)
    except Exception as exc:  # noqa: BLE001
        return Check("GADM host reachable", FAIL, str(exc)[:200], "Check your network or proxy")
    ok = resp.status_code < 500
    return Check(
        "GADM host reachable",
        PASS if ok else FAIL,
        f"HTTP {resp.status_code}",
        "" if ok else "Try geoBoundaries instead, see config boundary_source",
    )


def run_all() -> list[Check]:
    checks: list[Check] = []

    env_check, project = check_env()
    checks.append(env_check)

    imp = check_import()
    checks.append(imp)
    if imp.status == FAIL:
        return checks

    checks.append(check_credentials())

    if project is None:
        return checks

    init = check_initialise(project)
    checks.append(init)
    if init.status == FAIL:
        return checks

    checks.append(check_compute())

    try:
        cfg = country_config()
        checks.append(check_viirs(cfg))
    except ConfigError as exc:
        checks.append(Check("Country config", FAIL, str(exc), "Check config/countries.yaml"))

    checks.append(check_boundaries_reachable())
    return checks


def main() -> int:
    checks = run_all()
    width = max(len(c.name) for c in checks) + 2
    print()
    for c in checks:
        print(f"[{c.status}] {c.name.ljust(width)} {c.detail}")
        if c.remedy:
            print(f"       {'':<{width}} fix: {c.remedy}")
    failed = [c for c in checks if c.status == FAIL]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed. Work through SETUP.md.")
        return 1
    print("Phase 0 complete. You are ready to pull boundaries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
