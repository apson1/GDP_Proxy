# Phase 0 Setup Guide

Follow this in order. Do not skip step 4, which is where most people get stuck.

---

## Part A: Google Earth Engine access

### Why people get stuck

Earth Engine access needs three separate things to be true, and the error messages do not tell you which one is missing:

1. A Google Cloud project exists.
2. The Earth Engine API is enabled on that project.
3. That project is registered with Earth Engine as commercial or noncommercial.

Since June 2024, an unregistered project fails even with the API enabled. Since April 2026, registration also assigns you a compute quota tier. If you tried before and it broke, step 3 is almost certainly the reason.

### Step 1: Use the right Google account

Use a personal Gmail or an academic account. Work accounts on Google Workspace often have Cloud project creation blocked by an admin policy, and the failure looks like a permissions error rather than a policy error. If you see "You do not have permission to create projects", that is the cause. Switch accounts.

Check which account you are in at https://console.cloud.google.com before doing anything else. Google silently uses whichever account is first in your browser session, and multi-account users end up creating the project under one account and authenticating with another.

### Step 2: Create the Cloud project

1. Go to https://console.cloud.google.com/projectcreate
2. Project name: `gdp-proxy`. Google generates a project ID like `gdp-proxy-472913`. Copy that ID. The ID, not the name, is what the code needs.
3. Leave organisation as "No organisation" if offered.
4. Click Create and wait for the notification bell to confirm.

No credit card is required for this step.

### Step 3: Enable the Earth Engine API

1. Go to https://console.cloud.google.com/apis/library/earthengine.googleapis.com
2. Confirm the project selector at the top shows `gdp-proxy`. This is the most common mistake, enabling the API on the wrong project.
3. Click Enable.
4. Wait two to three minutes. The enablement propagates asynchronously, and authenticating immediately after can still fail.

### Step 4: Register the project with Earth Engine

This is the step that is missing from most tutorials.

1. Go to https://code.earthengine.google.com/register
2. Select your `gdp-proxy` project.
3. Choose "Unpaid usage" and then the noncommercial purpose that matches you: academic research, education, or nonprofit. Pick honestly. Misdeclaring is a terms violation and they do audit.
4. Submit. Access is usually immediate. Some noncommercial declarations get manually reviewed and take up to a few days. You will get an email either way.

Confirm at https://code.earthengine.google.com. If the Code Editor loads and lets you run a script, you are registered.

### Step 5: Get the Contributor tier (free, worth doing)

The default Community tier gives 150 EECU-hours per month. Contributor gives 1000. Contributor requires a billing account attached to the project, but noncommercial usage is not charged.

1. Go to https://console.cloud.google.com/billing
2. Create a billing account. This needs a card. You will not be billed for noncommercial Earth Engine use.
3. Link it to `gdp-proxy`.
4. To be safe, set a budget alert at a low value like 1 USD so you get an email if anything ever does bill.

If you are not comfortable putting a card on file, skip this. 150 EECU-hours is enough for a single-country ADM2 project if you follow the batch-export discipline in CLAUDE.md. It is not enough if you iterate carelessly with interactive calls.

Check your usage any time at https://console.cloud.google.com/earth-engine

### Step 6: Authenticate locally

```bash
pip install earthengine-api
earthengine authenticate
```

This opens a browser. Sign in with the same account from step 1. Approve the scopes. It writes credentials to `~/.config/earthengine/credentials` on Linux and macOS, or `%USERPROFILE%\.config\earthengine\credentials` on Windows.

Then set the project as default:

```bash
earthengine set_project gdp-proxy-472913
```

Verify:

```bash
python -c "import ee; ee.Initialize(project='gdp-proxy-472913'); print(ee.Number(1).add(1).getInfo())"
```

It should print `2`.

### Step 7: Run the project doctor

```bash
python -m gdp_proxy.doctor
```

This checks auth, project registration, dataset reachability, and prints the newest available VIIRS month. That is your Phase 0 exit test.

---

## Error reference

| Error text | Cause | Fix |
|---|---|---|
| `Not signed up for Earth Engine or project is not registered` | Step 4 was skipped, or you are initialising against a different project than the one you registered | Register at code.earthengine.google.com/register, then pass the exact project ID to `ee.Initialize(project=...)` |
| `Earth Engine API has not been used in project X before or it is disabled` | API not enabled, or enabled on the wrong project | Enable at the API library URL in step 3 with the correct project selected, wait 3 minutes |
| `Caller does not have required permission` | The authenticated account is not an owner or editor on the project | Add the account under IAM with the Earth Engine Resource Viewer plus Service Usage Consumer roles, or authenticate with the owning account |
| `ee.Initialize: no project found` | No default project set and none passed in code | `earthengine set_project <id>`, or always pass `project=` explicitly |
| `Token has been expired or revoked` | Stale cached credentials | `earthengine authenticate --force` |
| `You do not have permission to create projects` | Workspace admin policy | Use a personal Gmail account |
| `Quota exceeded` or jobs suddenly crawling | You hit the monthly EECU limit and are in Restricted mode | Check the Earth Engine usage page, switch to batch exports, wait for the monthly reset, or move to Contributor tier |

Two habits that prevent most of this: always pass `project=` explicitly in code rather than relying on a default, and keep the project ID in `.env` so there is one source of truth.

---

## Part B: NASA Earthdata (fallback and Black Marble)

1. Register at https://urs.earthdata.nasa.gov/users/new. Free, instant, no approval.
2. Log in to https://ladsweb.modaps.eosdis.nasa.gov, open your profile, and generate an App Key. This is the LAADS token for downloading VNP46 products directly.
3. Put the username, password, and token in `.env`.

You only need this if you leave GEE or want Black Marble products that GEE does not carry.

---

## Part C: EOG gas flare licence

Apply now, because approval is not instant.

1. Register at https://eogdata.mines.edu/products/register/
2. Submit the VIIRS Nightfire data use licence application describing your research use.
3. Interim access only gives the reduced ezCSV format. Full CSV needs the approved licence.

While waiting, use the published global flare survey point locations to build a manual mask for the petroleum basins in your target country.

---

## Part E: Upload boundaries as an Earth Engine asset

Do this once, after Phase 1 produces a boundary snapshot. Sending several
hundred detailed district polygons inline with every request will hit the
request size limit, and the error does not tell you that is the cause.

1. Export the snapshot to a shapefile:

   ```python
   import geopandas as gpd

   gdf = gpd.read_parquet("data/processed/boundaries_india_adm2_gadm.parquet")
   gdf[["region_id", "name", "geometry"]].to_file("data/interim/boundaries_india_adm2.shp")
   ```

2. Open https://code.earthengine.google.com, go to Assets, New, Shapefile.
3. Upload the .shp, .shx, .dbf and .prj together. Name it `boundaries_india_adm2`.
4. Wait for the ingestion task to finish, then copy the asset ID.
5. Put it in `config/countries.yaml` under `boundary_asset`, for example
   `projects/gdp-proxy-472913/assets/boundaries_india_adm2`.

Keep `region_id` in the uploaded attributes. It is the only join key the rest
of the pipeline uses.

---

## Part D: Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Windows note: `geopandas` and `rasterio` used to be painful to install with pip on Windows. Recent wheels work, but if you hit a GDAL build error, use conda instead:

```bash
conda create -n gdp-proxy python=3.11
conda activate gdp-proxy
conda install -c conda-forge geopandas rasterio pyproj shapely
pip install -e ".[dev]"
```

---

## Setup checklist

- [ ] Cloud project created, project ID copied
- [ ] Earth Engine API enabled on that exact project
- [ ] Project registered as noncommercial at code.earthengine.google.com/register
- [ ] Code Editor loads and runs a script
- [ ] Billing attached for Contributor tier, or accepted the 150 EECU-hour limit
- [ ] `earthengine authenticate` succeeded
- [ ] `.env` filled in from `.env.example`
- [ ] `python -m gdp_proxy.doctor` passes and prints the newest VIIRS month
- [ ] Earthdata account created and LAADS token saved
- [ ] EOG licence application submitted
