# Deployment and Maintenance

This guide covers the production Railway + Vercel setup and the operational issues that are
easy to forget between deployments.

## Services

- Railway API service: builds the root Dockerfile and runs `uvicorn api.main:app`.
- Railway PostgreSQL service: stores prediction history and model registry metadata.
- Vercel web project: serves the Next.js dashboard and proxies API calls to Railway.

## Required Environment Variables

### Railway API

```dotenv
DATABASE_URL=${{Postgres.DATABASE_URL}}
LOG_LEVEL=INFO
JSON_LOGS=false
```

Use Railway's variable reference if the database service is named `Postgres`. If the service
has another name, use that name in the reference.

Do not set this in production unless you intentionally want to skip migrations:

```dotenv
SOLPREDICT_SKIP_MIGRATIONS=1
```

### Vercel

```dotenv
SOLPREDICT_API_URL=https://<railway-api-public-domain>
```

Use the Railway API base URL only. These are wrong:

```dotenv
SOLPREDICT_API_URL=https://<railway-api-public-domain>/predict
SOLPREDICT_API_URL=https://<vercel-domain>
```

`NEXT_PUBLIC_API_URL` is optional and redundant for the current production proxy setup.

## Deploy Order

1. Merge changes into `main`.
2. Redeploy the Railway API from `main`.
3. Confirm `/health` returns both models loaded.
4. Redeploy the Vercel dashboard from `main`.
5. Run the smoke tests below.

## Smoke Tests

Test Railway directly:

```bash
curl https://<railway-api-public-domain>/health
curl https://<railway-api-public-domain>/models
curl "https://<railway-api-public-domain>/history?limit=10&offset=0"
```

Test through Vercel:

```bash
curl https://<vercel-domain>/models
curl "https://<vercel-domain>/history?limit=10&offset=0"
curl -X POST https://<vercel-domain>/predict \
  -H "content-type: application/json" \
  --data '{"smiles":"CCO"}'
```

`/predict` should include both `random_forest` and `neural_network` predictions.

## Fresh Database Model Registry Seed

Alembic creates the tables, but it does not register model-version rows. If Railway
PostgreSQL is new, `/models` will return `[]` until rows are inserted or the training
pipeline is run against that database.

For the bundled production artifacts, seed active rows with:

```sql
INSERT INTO model_versions (
  name,
  version,
  artifact_path,
  mlflow_run_id,
  trained_at,
  cv_r2_mean,
  cv_rmse_mean,
  test_r2,
  test_rmse,
  hyperparameters,
  is_active
)
VALUES
(
  'random_forest',
  'railway-bundled-rf',
  'models/random_forest.pkl',
  NULL,
  NOW(),
  NULL,
  NULL,
  NULL,
  NULL,
  '{}'::json,
  TRUE
),
(
  'neural_network',
  'railway-bundled-nn',
  'models/neural_network.pt',
  NULL,
  NOW(),
  NULL,
  NULL,
  NULL,
  NULL,
  '{}'::json,
  TRUE
);
```

After seeding, restart or redeploy the Railway API so startup loads the active rows. New
predictions will then include model-version tags. Existing history rows created before
seeding can still show `n/a`.

`No metrics available yet` is expected for these manual rows because the metric columns are
`NULL`. Run `scripts/train.py` against the production database or update the metric columns
manually if production cards should show RMSE/CV metrics.

## Troubleshooting

### Vercel `/models`, `/history`, or `/predict` returns 404

Cause: the deployed Vercel build does not include the Next.js proxy routes, or Vercel is
serving an old deployment.

Fix:

1. Confirm `main` includes `web/src/app/models/route.ts`, `history/route.ts`, and
   `predict/route.ts`.
2. Redeploy Vercel from `main`.

### Vercel returns 502 from `/models`, `/history`, or `/predict`

Cause: the proxy route exists, but cannot reach the Railway API.

Fix:

1. Check Vercel `SOLPREDICT_API_URL`.
2. Confirm it is the Railway API base URL.
3. Confirm `curl https://<railway-api-public-domain>/health` works.

### Active model cards show `Unavailable`

Cause: `/models` returned no active `model_versions` rows.

Fix: seed `model_versions` using the SQL above, or run `python3 scripts/train.py` with
`DATABASE_URL` pointed at the production database.

### History model-version badges show `n/a`

Cause: the prediction row was created before active model-version rows existed or before the
API was restarted after seeding.

Fix: seed model-version rows, restart the API, and create a new prediction. Old rows can
remain `n/a`.

### Railway crashes with `ModuleNotFoundError: No module named 'psycopg2'`

Cause: the API is using a PostgreSQL `DATABASE_URL` without the Postgres DBAPI package.

Fix: deploy a commit that includes `psycopg2-binary` in `pyproject.toml` or
`api/requirements.txt`.

### Railway image push is slow

Cause: the API image contains scientific Python dependencies and bundled model artifacts.
PyTorch, RDKit, scikit-learn, and model files make the image large.

Fix: wait for the push unless it is clearly stalled. If it repeatedly stalls or costs too
much, reduce the runtime image by separating training-only dependencies from API inference
dependencies.
