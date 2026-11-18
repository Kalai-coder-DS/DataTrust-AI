# Deploying the public demo (Render, free plan)

This puts DataTrust AI online as an **open demo**: anyone with the link can try it without
logging in. You get a link like `https://datatrust-ai.onrender.com` for your CV and LinkedIn.
Nothing changes on your own computer. The local app works exactly as before.

---

## 1. What the demo is (and isn't)

| | On your laptop | On the public demo |
|---|---|---|
| Who can open it | only you (localhost) | anyone with the link |
| Login | optional password | none (open demo) |
| Database | MySQL or SQLite | SQLite inside the server's files |
| Who sees an upload | you | **only the browser that uploaded it** |
| File limits | 200 MB | 2 MB, 20,000 rows, 50 columns |
| Uploads | unlimited | 10 per hour per visitor (60 per hour for the whole server) |
| Summaries | template, Ollama or Claude | template only (no AI server online) |
| Model check | 3 cross-validation repeats, 5 importance shuffles | 1 repeat, 2 shuffles (slow server) |
| Uploads kept | for ever | up to 24 hours, and always lost on a restart |

The two sample datasets (`customers_dirty.csv`, `customers_clean.csv`) are always there for
every visitor, with the same numbers as on your laptop and in the README.

## 2. Data persistence: read this before you share the link

Render's free plan has an **ephemeral disk**. Every change made while the server runs
(visitor uploads, their database rows) is lost when the server:

- **sleeps:** after 15 minutes without visitors (the next visitor waits about a minute while it wakes up);
- **restarts** or **redeploys** (every `git push` redeploys).

The demo is built around this:

1. The **build step** runs `python scripts/seed_demo.py`, which analyses the two samples and
   saves them in `data/datatrust.db`. Files created at build time are part of the deployed
   service, so after **every** restart the database is back to "samples only".
2. At start-up, `wsgi.py` checks the samples again in a background thread (it adds them only
   if they are missing, so you never get duplicates) and deletes visitor uploads older than 24 hours.
3. Each upload also triggers that clean-up, and only the newest 300 visitor runs are kept.

So: **the demo is for trying the tool, not for storing results.** A visitor who wants to keep
a result downloads the HTML report or the JSON from the run page. The banner on every page
says this.

For storage that survives restarts you would need a paid disk or an external database. The
app already supports MySQL (`DATATRUST_DB=mysql`), so it would be a settings change, not a code change.

## 3. Security: how an open demo stays safe

- **Private uploads without accounts.** Every browser gets a random visitor id in a signed
  cookie (`datatrust_visitor`: HMAC-SHA256, HttpOnly, Secure, SameSite=Lax). Runs are saved
  with an owner tag derived from that id, in a separate table, `run_owners`. Pages show public
  runs plus your own. Another visitor's run, report, JSON, dataset history or job page gives the
  **same 404 as a page that doesn't exist**, so trying run numbers reveals nothing. The cookie
  value is never stored in the database.
- **Strict upload checks:**
  - the size limit is enforced twice: `413` before Django reads the request body, and again while the file is saved;
  - `.csv` only;
  - text only (binary files renamed to `.csv` are refused);
  - row and column limits;
  - rate limits per visitor, per address and per server;
  - one running analysis per visitor, and at most 3 queued for the whole server.
- **No secrets in GitHub.** The only secret is `DATATRUST_SECRET_KEY`. Render generates it
  (`generateValue: true` in `render.yaml`) and keeps it in its dashboard, so you never type it
  and it never appears in a file. `.env` and `*.db` stay in `.gitignore`.
- **HTTPS everywhere.**
  - HTTP is redirected to HTTPS, and browsers are told to stay on HTTPS for 30 days (HSTS).
  - The CSRF and visitor cookies are only sent over HTTPS.
  - The forwarded "https" header is trusted only in production behind Render's proxy.
- **Production checks.**
  - The app refuses to start with `DEBUG` on in demo mode, or with a missing or weak secret key.
  - `manage.py check --deploy` passes with no warnings (a test checks this).
- Everything from earlier phases still applies: escaped output, a strict Content-Security-Policy, and CSRF on every form.

## 4. Step by step

### Before you start
- The deployment patch is applied, all tests pass, and it's pushed to GitHub (`main`).
- You need nothing else: no card, no password, no API key. If Render asks for a card for the
  free plan, stop. Plans change, so check render.com/pricing.

### Steps
1. Open **https://render.com** and click **Get Started**, then **GitHub**. Sign in with your GitHub account.
2. Render asks for access to your repositories. Choose **Only select repositories**, pick
   **DataTrust-AI**, then click **Install** (or **Authorize**).
3. In the Render dashboard click **New +**, then **Blueprint**.
4. Select the **DataTrust-AI** repository and click **Connect**.
5. Render reads `render.yaml` and shows one service: **datatrust-ai** (web service, Free).
   - Blueprint name: type `datatrust-ai`.
   - Branch: `main`.
   - There is nothing to fill in. Every setting is in `render.yaml`, and the secret key is generated.
6. Click **Apply** (or **Deploy Blueprint**).
7. Open the service and watch **Logs**. The first build takes about 5 to 10 minutes:
   - `pip install ...` (pandas, scikit-learn, Django, gunicorn);
   - `customers_dirty.csv: added` and `customers_clean.csv: added` (the samples);
   - `Listening at: http://0.0.0.0:10000` and then **Your service is live**.
8. Click the link at the top (`https://datatrust-ai.onrender.com`, or a variant with a suffix
   if that name was taken). You should see the yellow **Public demo** banner and the two samples.

### Check it like a visitor
- Open a sample, then **Full report**. It should show Trust 79.8 for `customers_dirty.csv`.
- **Analyse a CSV**: upload a small CSV. You see "Analysing...", and then your result.
- Open the site in a **private window**. Your upload is not there; only the samples are.
  That is the privacy working.

### Updating later
Push to `main`; Render redeploys by itself. (That also wipes visitor uploads; see section 2.)

### Turning it off
In Render, open the service, go to **Settings** and click **Suspend** (or **Delete**). Your GitHub repository is not touched.

## 5. Settings reference (all in `render.yaml`)

| Variable | Demo value | Meaning |
|---|---|---|
| `DATATRUST_DEBUG` | `0` | production mode |
| `DATATRUST_PUBLIC_DEMO` | `1` | open demo: no login, private uploads, limits |
| `DATATRUST_SECRET_KEY` | generated by Render | signs cookies; never in git |
| `DATATRUST_DB` / `DATATRUST_SQLITE_PATH` | `sqlite` / `data/datatrust.db` | database |
| `DATATRUST_MAX_FILE_MB` | `2` | upload size limit |
| `DATATRUST_DEMO_MAX_ROWS` / `_MAX_COLUMNS` | `20000` / `50` | per file |
| `DATATRUST_DEMO_UPLOADS_PER_HOUR` | `10` | per visitor and per address |
| `DATATRUST_DEMO_UPLOADS_PER_HOUR_TOTAL` | `60` | whole server |
| `DATATRUST_DEMO_MAX_PENDING` | `3` | analyses queued or running, whole server |
| `DATATRUST_DEMO_RETENTION_HOURS` | `24` | visitor uploads deleted after this |
| `DATATRUST_DEMO_MAX_STORED_RUNS` | `300` | oldest visitor runs deleted beyond this |
| `DATATRUST_ML_MAX_ROWS` / `_CV_REPEATS` / `_IMPORTANCE_REPEATS` | `3000` / `1` / `2` | lighter model checks |
| `DATATRUST_WEB_WORKERS` | `1` | one analysis at a time |
| `RENDER_EXTERNAL_HOSTNAME` | set by Render | added to `ALLOWED_HOSTS` automatically |

To change a limit, edit `render.yaml` and push. Changing it only in the Render dashboard also
works, but `render.yaml` is the record of what runs.

## 6. Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| Build fails at `pip install` with "No matching distribution" | the Python version isn't available | check `PYTHON_VERSION` in `render.yaml` (a full version, e.g. `3.13.5`); click **Manual Deploy**, then **Clear build cache & deploy** |
| Build log: `ModuleNotFoundError: gunicorn` at start | old `requirements.txt` without gunicorn | make sure the patch is pushed; redeploy |
| Log: `ImproperlyConfigured: ... DATATRUST_SECRET_KEY` | the secret wasn't generated (service created by hand, not as a Blueprint) | **Environment**, then add `DATATRUST_SECRET_KEY` with a random value of 50+ characters (**Generate**) |
| `Bad Request (400)` on every page | the hostname isn't allowed | Render normally sets `RENDER_EXTERNAL_HOSTNAME` itself; for a custom domain add it to `DATATRUST_ALLOWED_HOSTS` |
| Stuck on Render's "waking up" page; UptimeRobot says Down; logs show `"HEAD / HTTP/1.1" 405` | page views allowed GET only, and these checks use HEAD | fixed in the "answer HEAD requests" patch: pages now answer HEAD with 200 |
| The page loads for about a minute | the free server was asleep | normal on the free plan; open the link a minute before showing it to someone |
| "Too many redirects" | `DATATRUST_HTTPS` is on, but the proxy header is missing (only outside Render) | on Render this doesn't happen; elsewhere set `DATATRUST_HTTPS=0` behind a proxy that doesn't send `X-Forwarded-Proto` |
| My upload disappeared | restart, sleep, redeploy, or older than 24 h | expected (section 2); download the report if you want to keep it |
| "This analysis is not known to the server" | the server restarted while the analysis ran | upload again |
| `429` "Upload limit reached" or "busy" | the demo limits | wait the time shown; the limits protect the free server |
| `413` "File too large" | over 2 MB | use a smaller file, or run the app locally |
| An upload shows in one browser but not another | the privacy rule | expected: each browser is a separate visitor |
| Service killed: "Out of memory" in the logs | a file near the limits on a 512 MB server | lower `DATATRUST_DEMO_MAX_ROWS` or `DATATRUST_ML_MAX_ROWS` in `render.yaml` |
| Render asks for a card | free-plan rules changed | stop; the app also runs on other hosts with the same `gunicorn` command |

Logs are in the Render dashboard under **Logs** (the app logs to the console there).
