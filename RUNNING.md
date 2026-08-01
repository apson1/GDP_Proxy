# Running the project

Two things people get stuck on. Both are environment problems, not code problems.

---

## 1. Starting the dashboard

The `streamlit` command lives inside the virtual environment. Typing `streamlit`
in a fresh terminal calls a program that is not on your PATH, which is why it
looks like the app is broken when it is not.

```bat
cd "C:\Git Repo\GDP_Proxy"
.venv\Scripts\activate
streamlit run app\streamlit_app.py
```

Your prompt should read `(.venv) C:\Git Repo\GDP_Proxy>` once the environment is
active. Streamlit prints a local URL, usually http://localhost:8501, and opens
your browser. Press Ctrl+C in the terminal to stop it.

If you would rather not activate the environment each time, call the executable
directly:

```bat
.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

### If something looks wrong

| Symptom | Cause | Fix |
|---|---|---|
| `'streamlit' is not recognized` | environment not active | run `.venv\Scripts\activate` first |
| Sidebar looks missing for a second | Streamlit mounts the main pane before the sidebar | wait, or refresh once |
| `FileNotFoundError` on estimates | pipeline has not produced output | `python -m gdp_proxy.pipeline` |
| Port already in use | an earlier run is still going | `streamlit run app\streamlit_app.py --server.port 8502` |
| Blank map | browser blocked the map tiles | check the browser console, try another browser |

Everything the app reads is precomputed. It never calls Earth Engine, so opening
it costs you no quota.

---

## 2. Setting up the scheduled check

`.github/workflows/check-new-month.yml` runs every Monday and asks Earth Engine
whether NOAA has posted a new VIIRS composite. It reads two repository secrets
that do not exist yet, so its first run will fail until you create them.

This takes about fifteen minutes. Steps 1 to 3 happen in the Google Cloud
console, step 4 in GitHub.

### Step 1: Create a service account

A service account is a robot identity. It is used instead of your personal login
because a personal refresh token sitting in a repository secret is a standing
credential that nobody ever rotates.

1. Go to https://console.cloud.google.com/iam-admin/serviceaccounts
2. Confirm the project selector at the top reads `gdp-proxy-472913`. This is the
   same trap as enabling the API on the wrong project.
3. Click **Create service account**.
4. Name it `gdp-proxy-ci`. Google fills in an email like
   `gdp-proxy-ci@gdp-proxy-472913.iam.gserviceaccount.com`. Copy that email, you
   need it in step 3.
5. Click **Create and continue**.
6. Grant it the role **Earth Engine Resource Viewer**. If you cannot find it,
   type "earth engine" in the role filter. Add **Service Usage Consumer** as
   well; without it you get a permissions error that does not mention either
   role.
7. Click **Done**.

### Step 2: Download a JSON key

1. Click the service account you just created.
2. Open the **Keys** tab.
3. **Add key**, **Create new key**, choose **JSON**, **Create**.
4. A `.json` file downloads. Treat it like a password. Anyone holding it can use
   your Earth Engine quota.

Do not put this file in the repository. `.gitignore` does not cover it and a
committed key is a bad afternoon.

### Step 3: Register the service account with Earth Engine

This is the step people miss, and the error message when you skip it does not
mention registration. Creating the account and granting it Earth Engine access
are two different things.

1. Go to https://code.earthengine.google.com/register
2. Choose the option to register a service account (not a new project).
3. Paste the service account email from step 1.
4. Submit.

If that page does not offer a service account option, use the Earth Engine
project settings at https://console.cloud.google.com/earth-engine and add the
service account email as a project member there instead.

### Step 4: Add the two GitHub secrets

1. Open your repository on github.com.
2. **Settings**, then **Secrets and variables** in the left sidebar, then
   **Actions**.
3. Click **New repository secret**, twice.

| Name | Value |
|---|---|
| `GEE_SERVICE_ACCOUNT_JSON` | the entire contents of the JSON file from step 2, opened in a text editor, copied whole including the outer braces |
| `GEE_PROJECT_ID` | `gdp-proxy-472913` |

The names must match exactly. The workflow reads them by name and a typo shows
up as an authentication failure rather than a missing-secret error.

### Step 5: Test it without waiting for Monday

1. Repository, **Actions** tab.
2. Select **Check for a new VIIRS month** in the left list.
3. **Run workflow**, then the green **Run workflow** button.
4. Watch the run. Green is a pass.

A successful run prints either that a new composite is available or that you are
up to date. Both are correct outcomes. Most weeks it will find nothing, which is
the intended behaviour.

### If the run fails

| Error | Meaning |
|---|---|
| `Not signed up for Earth Engine or project is not registered` | step 3 was skipped or has not propagated |
| `Caller does not have required permission` | the service account is missing a role from step 1 |
| `Could not parse credentials` | the JSON secret is truncated; paste the whole file including braces |
| `GEE_PROJECT_ID is not set` | secret missing or misspelled |

### If you would rather not set this up yet

Disable the workflow rather than leaving it failing every Monday. Actions tab,
select the workflow, the `...` menu on the right, **Disable workflow**. A
repository with a permanently red badge is one you stop reading.

---

## 3. What the schedule does and does not do

The cron only *checks*. It does not extract, because an Earth Engine export
writes CSVs to your Google Drive and a human has to pull them down. Running
extraction unattended on a schedule is the fastest way to spend a month of EECU
quota on a bug.

When the check reports a new month, the manual sequence is:

```bat
.venv\Scripts\activate
python -m gdp_proxy.extract --series annual --years <YEAR> --submit
python -m gdp_proxy.extract --status
REM download the CSVs from the gdp_proxy_exports folder in Google Drive
REM into data\raw\exports\ , then:
python -m gdp_proxy.extract --series annual --years <YEAR> --ingest
python -m gdp_proxy.features
python -m gdp_proxy.pipeline
python -m pytest -q
```
