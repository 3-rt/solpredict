# SolPredict Dashboard

Next.js 16 single-page dashboard for the SolPredict API. Renders the predict form, prediction history, model registry, and model-comparison charts.

## Running locally

The dashboard expects the API on `http://localhost:7860` (see the root README for starting it).

```bash
npm install
npm run dev
```

Override the API origin with `NEXT_PUBLIC_API_URL`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:7860 npm run dev
```

## Layout

All sections live in `src/app/page.tsx` as a single route:

- **Predict** — SMILES input, example pills, prediction cards, descriptor grid
- **History** — active-model strip, recent-prediction distribution chart (Recharts), paginated history table backed by `/history` and `/models`
- **Model Comparison** — static metrics sourced from `data/results.json`
- **Methodology / footer**

## Scripts

```bash
npm run dev     # local dev server on :3000
npm run build   # production build
npm run lint    # eslint (flat config)
```

## Stack

- Next.js 16 App Router, React 19
- Tailwind CSS 4
- Recharts for charts

## Related docs

- [../README.md](../README.md) — full-stack quick start
- [../docs/architecture.md](../docs/architecture.md) — system overview
- [../docs/api-reference.md](../docs/api-reference.md) — endpoints the dashboard consumes
