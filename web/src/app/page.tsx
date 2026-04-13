"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";
const HISTORY_PAGE_SIZE = 10;

const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Glucose", smiles: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
];

const SOLUBILITY_BANDS = [
  "Highly soluble",
  "Soluble",
  "Moderately soluble",
  "Sparingly soluble",
  "Poorly soluble",
  "Practically insoluble",
] as const;

const TEAL = "#1B4965";
const SIENNA = "#B74530";

const CHART_TOOLTIP = {
  contentStyle: {
    background: "#fff",
    border: "1px solid #e5e2db",
    borderRadius: 4,
    color: "#2d2d2d",
    fontSize: 12,
  },
};

interface PredictionResult {
  smiles: string;
  valid: boolean;
  predictions?: { random_forest?: number; neural_network?: number };
  descriptors?: {
    molecular_weight: number;
    logp: number;
    hbd: number;
    hba: number;
    tpsa: number;
  };
  molecule_name?: string | null;
  error?: string;
}

interface Results {
  models: {
    random_forest: { test_metrics: { r2: number; rmse: number; mae: number } };
    neural_network: { test_metrics: { r2: number; rmse: number; mae: number } };
  };
  plots: {
    scatter: { y_true: number[]; rf_pred: number[]; nn_pred: number[] };
    residuals: { rf_residuals: number[]; nn_residuals: number[] };
  };
}

interface HistoryItem {
  id: number;
  smiles: string;
  molecule_name: string | null;
  rf_prediction: number | null;
  nn_prediction: number | null;
  descriptors: {
    molecular_weight?: number;
    logp?: number;
    hbd?: number;
    hba?: number;
    tpsa?: number;
  };
  created_at: string;
  rf_model_version: string | null;
  nn_model_version: string | null;
}

interface HistoryResponse {
  items: HistoryItem[];
  total: number;
}

interface ModelVersion {
  id: number;
  name: string;
  version: string;
  mlflow_run_id: string | null;
  artifact_path: string;
  trained_at: string | null;
  cv_r2_mean: number | null;
  cv_rmse_mean: number | null;
  test_r2: number | null;
  test_rmse: number | null;
  hyperparameters: Record<string, unknown>;
  is_active: boolean;
}

export default function Home() {
  return (
    <div className="px-4 pb-12 pt-24 sm:px-6 sm:pb-16 sm:pt-32">
      <div className="mx-auto max-w-2xl">
        <PredictSection />
      </div>
      <Divider />
      <HistorySection />
      <Divider />
      <div className="mx-auto max-w-2xl">
        <ComparisonSection />
      </div>
      <Divider />
      <div className="mx-auto max-w-2xl">
        <MethodologySection />
      </div>
      <Divider />
      <div className="mx-auto max-w-2xl">
        <Footer />
      </div>
    </div>
  );
}

function Divider() {
  return <hr className="my-20 border-t border-[var(--border)]" />;
}

function describeSolubility(logS: number): { label: string; description: string } {
  if (logS >= 0) return { label: "Highly soluble", description: "Dissolves very easily in water" };
  if (logS >= -1) return { label: "Soluble", description: "Dissolves readily in water" };
  if (logS >= -2) {
    return { label: "Moderately soluble", description: "Dissolves with some difficulty" };
  }
  if (logS >= -4) return { label: "Sparingly soluble", description: "Dissolves poorly in water" };
  if (logS >= -6) {
    return { label: "Poorly soluble", description: "Very limited dissolution in water" };
  }
  return { label: "Practically insoluble", description: "Essentially does not dissolve in water" };
}

function formatPrediction(value: number | null | undefined): string {
  return value == null ? "\u2014" : value.toFixed(2);
}

function formatTimestamp(value: string | null): string {
  if (!value) return "Unknown";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function modelLabel(name: string): string {
  return name === "random_forest" ? "Random Forest" : "Neural Network";
}

function activeModelCard(model: ModelVersion | undefined) {
  if (!model) {
    return {
      value: "Unavailable",
      detail: "No active version registered yet",
      metrics: [],
    };
  }

  const metrics = [
    model.test_rmse != null ? `Test RMSE ${model.test_rmse.toFixed(2)}` : null,
    model.cv_r2_mean != null ? `CV R² ${model.cv_r2_mean.toFixed(2)}` : null,
  ].filter((value): value is string => value !== null);

  return {
    value: model.version,
    detail: model.trained_at ? `Trained ${formatTimestamp(model.trained_at)}` : "Training time unknown",
    metrics,
  };
}

function buildDistributionData(items: HistoryItem[]) {
  const totals = new Map<string, { label: string; RF: number; NN: number }>();

  for (const label of SOLUBILITY_BANDS) {
    totals.set(label, { label, RF: 0, NN: 0 });
  }

  for (const item of items) {
    if (item.rf_prediction != null) {
      const row = totals.get(describeSolubility(item.rf_prediction).label);
      if (row) row.RF += 1;
    }
    if (item.nn_prediction != null) {
      const row = totals.get(describeSolubility(item.nn_prediction).label);
      if (row) row.NN += 1;
    }
  }

  return SOLUBILITY_BANDS.map((label) => totals.get(label) ?? { label, RF: 0, NN: 0 });
}

function VersionBadge({ label }: { label: string | null }) {
  return (
    <span className="rounded-full border border-[var(--border)] bg-[var(--bg-code)] px-2.5 py-1 text-[10px] text-[var(--text-secondary)]">
      {label ?? "n/a"}
    </span>
  );
}

function PredictSection() {
  const [smiles, setSmiles] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict(input?: string) {
    const smilesInput = (input || smiles).trim();
    if (!smilesInput) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: smilesInput }),
      });
      const data: PredictionResult = await res.json();
      setResult(data);
      if (!data.valid) setError(data.error || "Invalid SMILES string");
    } catch {
      setError("Could not reach the prediction server. It may be waking up — try again in 30s.");
    } finally {
      setLoading(false);
    }
  }

  function handleExample(mol: { name: string; smiles: string }) {
    setSmiles(mol.smiles);
    void handlePredict(mol.smiles);
  }

  return (
    <section>
      <h1 className="mb-3 text-2xl font-bold">SolPredict</h1>
      <p className="mb-14 leading-relaxed text-[var(--text-secondary)]">
        Predict aqueous solubility of organic molecules using Random Forest and Neural Network
        models trained on the{" "}
        <a
          href="https://pubs.acs.org/doi/10.1021/ci034243x"
          target="_blank"
          rel="noopener noreferrer"
          className="underline transition-colors hover:text-[var(--text)]"
        >
          ESOL dataset
        </a>
        . Enter a SMILES string or try one of the examples.
      </p>

      <div className="mb-3 flex gap-2">
        <input
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void handlePredict()}
          placeholder="Enter SMILES (e.g. CCO)"
          className="flex-1 rounded-lg border border-[var(--border)] bg-white px-4 py-2.5 text-sm font-[family-name:var(--font-mono)] placeholder:text-[var(--text-muted)] transition-colors focus:border-[var(--teal)] focus:outline-none"
        />
        <button
          onClick={() => void handlePredict()}
          disabled={loading || !smiles.trim()}
          className="rounded-lg bg-[var(--teal)] px-5 py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {loading ? "..." : "Predict"}
        </button>
      </div>

      <div className="mb-14 flex flex-wrap gap-1.5">
        {EXAMPLES.map((mol) => (
          <button
            key={mol.name}
            onClick={() => handleExample(mol)}
            className="rounded bg-[var(--bg-code)] px-2.5 py-1 text-xs text-[var(--text-secondary)] transition-colors hover:text-[var(--text)]"
          >
            {mol.name}
          </button>
        ))}
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-[var(--sienna)]">
          {error}
        </div>
      )}

      {result && result.valid && (
        <div>
          {result.molecule_name && (
            <p className="mb-5 text-sm text-[var(--text-secondary)]">
              Results for{" "}
              <span className="font-semibold text-[var(--text)]">{result.molecule_name}</span>
            </p>
          )}

          <div className="mb-6 grid gap-4 sm:grid-cols-2">
            {[
              {
                label: "Random Forest",
                value: result.predictions?.random_forest,
                color: TEAL,
                tag: "sklearn",
              },
              {
                label: "Neural Network",
                value: result.predictions?.neural_network,
                color: SIENNA,
                tag: "PyTorch",
              },
            ].map((m) => {
              const interp = m.value != null ? describeSolubility(m.value) : null;
              return (
                <div key={m.label} className="rounded-lg border border-[var(--border)] bg-white p-5">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-xs font-semibold" style={{ color: m.color }}>
                      {m.label}
                    </span>
                    <span className="font-[family-name:var(--font-mono)] text-[10px] text-[var(--text-muted)]">
                      {m.tag}
                    </span>
                  </div>
                  <div className="mb-0.5 font-[family-name:var(--font-mono)] text-2xl font-medium">
                    {m.value?.toFixed(2) ?? "\u2014"}
                  </div>
                  <div className="mb-3 text-xs text-[var(--text-muted)]">log mol/L</div>
                  {interp && (
                    <div className="border-t border-[var(--border)] pt-3">
                      <div className="text-sm font-medium">{interp.label}</div>
                      <div className="text-xs text-[var(--text-muted)]">{interp.description}</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {result.descriptors && (
            <div className="rounded-lg border border-[var(--border)] bg-white p-5">
              <div className="mb-3 text-xs font-medium text-[var(--text-muted)]">
                Molecular Descriptors
              </div>
              <div className="grid grid-cols-5 gap-4 text-center">
                {[
                  { value: result.descriptors.molecular_weight, label: "MW" },
                  { value: result.descriptors.logp, label: "LogP" },
                  { value: result.descriptors.hbd, label: "HBD" },
                  { value: result.descriptors.hba, label: "HBA" },
                  { value: result.descriptors.tpsa, label: "TPSA" },
                ].map((d) => (
                  <div key={d.label}>
                    <div className="font-[family-name:var(--font-mono)] text-base font-medium">
                      {d.value}
                    </div>
                    <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">{d.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function HistorySection() {
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API_URL}/history?limit=${HISTORY_PAGE_SIZE}&offset=${offset}`, {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<HistoryResponse>;
      })
      .then(setHistory)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setHistoryError(error instanceof Error ? error.message : "Failed to load history.");
      })
      .finally(() => setHistoryLoading(false));

    return () => controller.abort();
  }, [offset]);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API_URL}/models`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ModelVersion[]>;
      })
      .then(setModels)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setModelsError(error instanceof Error ? error.message : "Failed to load model metadata.");
      })
      .finally(() => setModelsLoading(false));

    return () => controller.abort();
  }, []);

  const activeModels = models.filter((model) => model.is_active);
  const activeRf = activeModels.find((model) => model.name === "random_forest");
  const activeNn = activeModels.find((model) => model.name === "neural_network");
  const historyItems = history?.items ?? [];
  const distributionData = buildDistributionData(historyItems);
  const total = history?.total ?? 0;
  const pageStart = total === 0 ? 0 : offset + 1;
  const pageEnd = Math.min(offset + HISTORY_PAGE_SIZE, total);
  const canPrevious = offset > 0;
  const canNext = offset + HISTORY_PAGE_SIZE < total;

  return (
    <section className="mx-auto max-w-6xl">
      <div className="mb-8 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 className="mb-2 text-xl font-bold">History</h2>
          <p className="max-w-2xl text-sm text-[var(--text-secondary)]">
            Browse recent predictions, see which model versions are active, and inspect how
            the latest outputs are distributed.
          </p>
        </div>
        <div className="rounded-full border border-[var(--border)] bg-white px-4 py-2 text-xs text-[var(--text-muted)]">
          Showing {pageStart}-{pageEnd} of {total || 0} recorded predictions
        </div>
      </div>

      <div className="mb-6 grid gap-4 lg:grid-cols-2">
        {[
          { key: "random_forest", color: TEAL, model: activeRf },
          { key: "neural_network", color: SIENNA, model: activeNn },
        ].map(({ key, color, model }) => {
          const summary = activeModelCard(model);
          return (
            <div key={key} className="rounded-2xl border border-[var(--border)] bg-white p-5">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color }}>
                    Active Model
                  </div>
                  <div className="mt-1 text-base font-semibold text-[var(--text)]">
                    {modelLabel(key)}
                  </div>
                </div>
                <span className="rounded-full bg-[var(--bg-code)] px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
                  {model ? "live" : "missing"}
                </span>
              </div>

              <div className="mb-1 font-[family-name:var(--font-mono)] text-lg font-medium text-[var(--text)]">
                {summary.value}
              </div>
              <div className="text-sm text-[var(--text-secondary)]">{summary.detail}</div>

              <div className="mt-4 flex flex-wrap gap-2">
                {summary.metrics.length > 0 ? (
                  summary.metrics.map((metric) => (
                    <span
                      key={metric}
                      className="rounded-full border border-[var(--border)] px-3 py-1 text-xs text-[var(--text-secondary)]"
                    >
                      {metric}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-[var(--text-muted)]">
                    {modelsLoading ? "Loading model metrics..." : "No metrics available yet"}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {modelsError && (
        <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-[var(--sienna)]">
          Could not load active model metadata.
        </div>
      )}

      <div className="mb-6 rounded-2xl border border-[var(--border)] bg-white p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold">Prediction Distribution</h3>
            <p className="text-sm text-[var(--text-secondary)]">
              Solubility bands across the current history page.
            </p>
          </div>
          <div className="text-xs text-[var(--text-muted)]">Recent {historyItems.length} rows</div>
        </div>

        {historyLoading ? (
          <p className="text-sm text-[var(--text-muted)]">Loading prediction distribution...</p>
        ) : historyError ? (
          <p className="text-sm text-[var(--sienna)]">Could not load prediction history.</p>
        ) : total === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">No predictions recorded yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={distributionData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
              <XAxis
                dataKey="label"
                tick={{ fill: "#666", fontSize: 11 }}
                stroke="#e5e2db"
                interval={0}
                angle={-18}
                textAnchor="end"
                height={70}
              />
              <YAxis allowDecimals={false} tick={{ fill: "#999", fontSize: 11 }} stroke="#e5e2db" />
              <Tooltip {...CHART_TOOLTIP} />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) => (value === "RF" ? "Random Forest" : "Neural Network")}
              />
              <Bar dataKey="RF" fill={TEAL} radius={[3, 3, 0, 0]} />
              <Bar dataKey="NN" fill={SIENNA} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="rounded-2xl border border-[var(--border)] bg-white p-5">
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-base font-semibold">Prediction History</h3>
            <p className="text-sm text-[var(--text-secondary)]">
              Recent molecules, timestamps, predictions, and model version tags.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                setHistoryLoading(true);
                setHistoryError(null);
                setOffset((value) => Math.max(0, value - HISTORY_PAGE_SIZE));
              }}
              disabled={!canPrevious}
              className="rounded-full border border-[var(--border)] px-4 py-2 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--teal)] hover:text-[var(--text)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Previous
            </button>
            <button
              type="button"
              onClick={() => {
                setHistoryLoading(true);
                setHistoryError(null);
                setOffset((value) => value + HISTORY_PAGE_SIZE);
              }}
              disabled={!canNext}
              className="rounded-full border border-[var(--border)] px-4 py-2 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--teal)] hover:text-[var(--text)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>

        {historyLoading ? (
          <p className="text-sm text-[var(--text-muted)]">Loading history...</p>
        ) : historyError ? (
          <p className="text-sm text-[var(--sienna)]">Could not load prediction history.</p>
        ) : total === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">No predictions recorded yet.</p>
        ) : (
          <>
            <div className="hidden overflow-x-auto lg:block">
              <table className="w-full min-w-[960px] text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] text-left text-[11px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
                    <th className="py-3 pr-4 font-medium">Molecule</th>
                    <th className="py-3 pr-4 font-medium">SMILES</th>
                    <th className="py-3 pr-4 font-medium">Logged</th>
                    <th className="py-3 pr-4 text-right font-medium">RF</th>
                    <th className="py-3 pr-4 text-right font-medium">NN</th>
                    <th className="py-3 font-medium">Model Versions</th>
                  </tr>
                </thead>
                <tbody>
                  {historyItems.map((item) => (
                    <tr key={item.id} className="border-b border-[var(--border)] align-top last:border-b-0">
                      <td className="py-4 pr-4">
                        <div className="font-medium text-[var(--text)]">
                          {item.molecule_name ?? "Unknown molecule"}
                        </div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          MW {item.descriptors.molecular_weight?.toFixed(2) ?? "\u2014"}
                        </div>
                      </td>
                      <td className="max-w-[260px] py-4 pr-4 font-[family-name:var(--font-mono)] text-xs text-[var(--text-secondary)]">
                        <span className="break-all">{item.smiles}</span>
                      </td>
                      <td className="py-4 pr-4 text-[var(--text-secondary)]">
                        {formatTimestamp(item.created_at)}
                      </td>
                      <td className="py-4 pr-4 text-right font-[family-name:var(--font-mono)] tabular-nums text-[var(--text)]">
                        {formatPrediction(item.rf_prediction)}
                      </td>
                      <td className="py-4 pr-4 text-right font-[family-name:var(--font-mono)] tabular-nums text-[var(--text)]">
                        {formatPrediction(item.nn_prediction)}
                      </td>
                      <td className="py-4">
                        <div className="flex flex-wrap gap-2">
                          <VersionBadge label={item.rf_model_version} />
                          <VersionBadge label={item.nn_model_version} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="grid gap-3 lg:hidden">
              {historyItems.map((item) => (
                <article key={item.id} className="rounded-xl border border-[var(--border)] bg-[var(--bg)] p-4">
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium text-[var(--text)]">
                        {item.molecule_name ?? "Unknown molecule"}
                      </div>
                      <div className="mt-1 text-xs text-[var(--text-muted)]">
                        {formatTimestamp(item.created_at)}
                      </div>
                    </div>
                    <div className="rounded-full bg-white px-3 py-1 text-[10px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
                      ID {item.id}
                    </div>
                  </div>

                  <div className="mb-3 break-all font-[family-name:var(--font-mono)] text-xs text-[var(--text-secondary)]">
                    {item.smiles}
                  </div>

                  <div className="mb-3 grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-white p-3">
                      <div className="mb-1 text-[10px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
                        Random Forest
                      </div>
                      <div className="font-[family-name:var(--font-mono)] text-lg text-[var(--text)]">
                        {formatPrediction(item.rf_prediction)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-white p-3">
                      <div className="mb-1 text-[10px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
                        Neural Network
                      </div>
                      <div className="font-[family-name:var(--font-mono)] text-lg text-[var(--text)]">
                        {formatPrediction(item.nn_prediction)}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <VersionBadge label={item.rf_model_version} />
                    <VersionBadge label={item.nn_model_version} />
                  </div>
                </article>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function ComparisonSection() {
  const [data, setData] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/results.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load results.");
      });
  }, []);

  if (error) {
    return (
      <section>
        <p className="text-sm text-[var(--sienna)]">Could not load model results.</p>
      </section>
    );
  }

  if (!data) {
    return (
      <section>
        <p className="text-sm text-[var(--text-muted)]">Loading results...</p>
      </section>
    );
  }

  const rfMetrics = data.models.random_forest.test_metrics;
  const nnMetrics = data.models.neural_network.test_metrics;

  const rfScatter = data.plots.scatter.y_true.map((y, i) => ({
    actual: y,
    predicted: data.plots.scatter.rf_pred[i],
  }));
  const nnScatter = data.plots.scatter.y_true.map((y, i) => ({
    actual: y,
    predicted: data.plots.scatter.nn_pred[i],
  }));

  const barData = [
    { metric: "R\u00B2", RF: rfMetrics.r2, NN: nnMetrics.r2 },
    { metric: "RMSE", RF: rfMetrics.rmse, NN: nnMetrics.rmse },
    { metric: "MAE", RF: rfMetrics.mae, NN: nnMetrics.mae },
  ];

  return (
    <section>
      <h2 className="mb-2 text-xl font-bold">Model Comparison</h2>
      <p className="mb-8 text-sm text-[var(--text-secondary)]">
        Side-by-side evaluation on the ESOL test set (20% holdout, n=226).
      </p>

      <div className="mb-10 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="py-2 pr-6 text-left font-medium text-[var(--text-muted)]">Metric</th>
              <th className="px-4 py-2 text-right font-semibold" style={{ color: TEAL }}>
                Random Forest
              </th>
              <th className="pl-4 py-2 text-right font-semibold" style={{ color: SIENNA }}>
                Neural Network
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              { label: "R\u00B2", rf: rfMetrics.r2, nn: nnMetrics.r2, higherBetter: true },
              { label: "RMSE", rf: rfMetrics.rmse, nn: nnMetrics.rmse, higherBetter: false },
              { label: "MAE", rf: rfMetrics.mae, nn: nnMetrics.mae, higherBetter: false },
            ].map((row) => {
              const rfWins = row.higherBetter ? row.rf >= row.nn : row.rf <= row.nn;
              return (
                <tr key={row.label} className="border-b border-[var(--border)]">
                  <td className="py-3 pr-6 text-[var(--text-secondary)]">{row.label}</td>
                  <td
                    className={`px-4 py-3 text-right font-[family-name:var(--font-mono)] tabular-nums ${rfWins ? "font-semibold" : ""}`}
                    style={rfWins ? { color: TEAL } : undefined}
                  >
                    {row.rf.toFixed(4)}
                  </td>
                  <td
                    className={`pl-4 py-3 text-right font-[family-name:var(--font-mono)] tabular-nums ${!rfWins ? "font-semibold" : ""}`}
                    style={!rfWins ? { color: SIENNA } : undefined}
                  >
                    {row.nn.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mb-10 grid gap-6 sm:grid-cols-2">
        {[
          { label: "Random Forest", color: TEAL, data: rfScatter },
          { label: "Neural Network", color: SIENNA, data: nnScatter },
        ].map(({ label, color, data: scatterData }) => (
          <div key={label}>
            <h3 className="mb-3 text-xs font-semibold" style={{ color }}>
              {label}{" "}
              <span className="font-normal text-[var(--text-muted)]">— Predicted vs Actual</span>
            </h3>
            <div className="rounded-lg border border-[var(--border)] bg-white p-3">
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
                  <XAxis
                    dataKey="actual"
                    name="Actual"
                    label={{
                      value: "Actual log(S)",
                      position: "insideBottom",
                      offset: -10,
                      fill: "#999",
                      fontSize: 10,
                    }}
                    tick={{ fill: "#999", fontSize: 10 }}
                    domain={[-10, 2]}
                    stroke="#e5e2db"
                  />
                  <YAxis
                    dataKey="predicted"
                    name="Predicted"
                    label={{
                      value: "Predicted",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#999",
                      fontSize: 10,
                    }}
                    tick={{ fill: "#999", fontSize: 10 }}
                    domain={[-10, 2]}
                    stroke="#e5e2db"
                  />
                  <Tooltip {...CHART_TOOLTIP} cursor={{ strokeDasharray: "3 3" }} />
                  <ReferenceLine
                    segment={[
                      { x: -10, y: -10 },
                      { x: 2, y: 2 },
                    ]}
                    stroke="#ccc"
                    strokeDasharray="4 4"
                  />
                  <Scatter data={scatterData} fill={color} fillOpacity={0.5} r={2.5} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>

      <div className="mb-10 grid gap-6 sm:grid-cols-2">
        {[
          { label: "Random Forest", color: TEAL, residuals: data.plots.residuals.rf_residuals },
          { label: "Neural Network", color: SIENNA, residuals: data.plots.residuals.nn_residuals },
        ].map(({ label, color, residuals }) => {
          const binCount = 20;
          const min = Math.min(...residuals);
          const max = Math.max(...residuals);
          const w = (max - min) / binCount;
          const bins = Array.from({ length: binCount }, (_, i) => {
            const lo = min + i * w;
            const hi = lo + w;
            const count = residuals.filter((r) => r >= lo && (i === binCount - 1 ? r <= hi : r < hi)).length;
            return { bin: lo.toFixed(1), count };
          });
          return (
            <div key={label}>
              <h3 className="mb-3 text-xs font-semibold" style={{ color }}>
                {label} <span className="font-normal text-[var(--text-muted)]">— Residuals</span>
              </h3>
              <div className="rounded-lg border border-[var(--border)] bg-white p-3">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={bins} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
                    <XAxis
                      dataKey="bin"
                      tick={{ fill: "#999", fontSize: 9 }}
                      label={{
                        value: "Residual",
                        position: "insideBottom",
                        offset: -10,
                        fill: "#999",
                        fontSize: 10,
                      }}
                      stroke="#e5e2db"
                    />
                    <YAxis tick={{ fill: "#999", fontSize: 10 }} stroke="#e5e2db" />
                    <Tooltip {...CHART_TOOLTIP} />
                    <Bar dataKey="count" fill={color} fillOpacity={0.6} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>

      <h3 className="mb-3 text-xs font-semibold text-[var(--text-muted)]">Metrics Overview</h3>
      <div className="rounded-lg border border-[var(--border)] bg-white p-3">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={barData} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
            <XAxis dataKey="metric" tick={{ fill: "#666", fontSize: 12 }} stroke="#e5e2db" />
            <YAxis tick={{ fill: "#999", fontSize: 11 }} stroke="#e5e2db" />
            <Tooltip {...CHART_TOOLTIP} />
            <Legend
              wrapperStyle={{ fontSize: 11 }}
              formatter={(value) => (value === "RF" ? "Random Forest" : "Neural Network")}
            />
            <Bar dataKey="RF" fill={TEAL} radius={[3, 3, 0, 0]} />
            <Bar dataKey="NN" fill={SIENNA} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function MethodologySection() {
  return (
    <section>
      <h2 className="mb-2 text-xl font-bold">Methodology</h2>
      <p className="mb-8 text-sm text-[var(--text-secondary)]">
        How SolPredict works, from data to predictions.
      </p>

      <h3 className="mb-2 text-base font-semibold">Dataset</h3>
      <p className="mb-4 text-sm leading-relaxed text-[var(--text-secondary)]">
        The{" "}
        <a
          href="https://pubs.acs.org/doi/10.1021/ci034243x"
          target="_blank"
          rel="noopener noreferrer"
          className="underline transition-colors hover:text-[var(--text)]"
        >
          ESOL dataset
        </a>{" "}
        (Delaney, 2004) contains experimentally measured aqueous solubility for 1,128 small
        organic molecules. Target variable is log(S) in mol/L. We use an 80/20 train/test
        split with random seed 42.
      </p>

      <h3 className="mb-2 mt-8 text-base font-semibold">Features</h3>
      <p className="mb-4 text-sm leading-relaxed text-[var(--text-secondary)]">
        Each molecule is converted to a 2048-bit Morgan fingerprint (ECFP4, radius=2) using
        RDKit. Each bit encodes the presence of a circular substructure, producing a
        fixed-length binary vector that standard ML algorithms can consume directly.
      </p>
      <div className="mb-4 rounded-lg bg-[var(--bg-code)] p-4 font-[family-name:var(--font-mono)] text-xs leading-relaxed">
        <div className="text-[var(--text-muted)]"># RDKit pipeline</div>
        <div>
          <span style={{ color: TEAL }}>from</span> rdkit{" "}
          <span style={{ color: TEAL }}>import</span> Chem
        </div>
        <div>
          <span style={{ color: TEAL }}>from</span> rdkit.Chem{" "}
          <span style={{ color: TEAL }}>import</span> AllChem
        </div>
        <div className="mt-2">
          mol = Chem.MolFromSmiles(
          <span style={{ color: SIENNA }}>&quot;CC(=O)Oc1ccccc1C(=O)O&quot;</span>)
        </div>
        <div>
          fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=
          <span style={{ color: SIENNA }}>2</span>, nBits=
          <span style={{ color: SIENNA }}>2048</span>)
        </div>
      </div>

      <h3 className="mb-3 mt-8 text-base font-semibold">Models</h3>
      <div className="mb-4 grid gap-6 sm:grid-cols-2">
        <div>
          <div className="mb-2 text-xs font-semibold" style={{ color: TEAL }}>
            Random Forest
          </div>
          <p className="mb-3 text-sm leading-relaxed text-[var(--text-secondary)]">
            Ensemble of 100 decision trees via bootstrap aggregation. Averages predictions to
            reduce variance.
          </p>
          <ul className="space-y-1 text-xs text-[var(--text-secondary)]">
            <li>
              <span className="text-[var(--text-muted)]">Library:</span> scikit-learn
            </li>
            <li>
              <span className="text-[var(--text-muted)]">Input:</span> 2048-bit fingerprint
            </li>
          </ul>
        </div>
        <div>
          <div className="mb-2 text-xs font-semibold" style={{ color: SIENNA }}>
            Neural Network
          </div>
          <p className="mb-3 text-sm leading-relaxed text-[var(--text-secondary)]">
            Feed-forward MLP (2048 {"\u2192"} 512 {"\u2192"} 128 {"\u2192"} 1) with ReLU and
            dropout (p=0.2). Trained with Adam, MSE loss, 100 epochs.
          </p>
          <ul className="space-y-1 text-xs text-[var(--text-secondary)]">
            <li>
              <span className="text-[var(--text-muted)]">Library:</span> PyTorch
            </li>
            <li>
              <span className="text-[var(--text-muted)]">LR:</span> 0.001, Batch: 64
            </li>
          </ul>
        </div>
      </div>

      <h3 className="mb-2 mt-8 text-base font-semibold">Evaluation</h3>
      <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
        Models are evaluated with <strong>R\u00B2</strong> (variance explained, higher is
        better), <strong>RMSE</strong> (root mean squared error, penalizes outliers), and{" "}
        <strong>MAE</strong> (mean absolute error, robust to outliers). All error metrics are
        in log mol/L units.
      </p>
    </section>
  );
}

function Footer() {
  const links = [
    { label: "Source Code", href: "https://github.com/3-rt/solpredict" },
    { label: "ESOL Paper", href: "https://pubs.acs.org/doi/10.1021/ci034243x" },
    { label: "Website", href: "https://www.basilliu.dev/" },
    { label: "GitHub", href: "https://github.com/3-rt" },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/basilliu/" },
    { label: "X", href: "https://x.com/basilliu_" },
  ];

  return (
    <footer className="flex items-center justify-between text-xs text-[var(--text-muted)]">
      <span>
        Built by{" "}
        <a
          href="https://www.basilliu.dev/"
          target="_blank"
          rel="noopener noreferrer"
          className="underline transition-colors hover:text-[var(--text)]"
        >
          Basil Liu
        </a>
      </span>
      <div className="flex flex-wrap gap-3">
        {links.map((link, i) => (
          <span key={link.label}>
            {i > 0 && <span className="mr-3">{"\u00B7"}</span>}
            <a
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="underline transition-colors hover:text-[var(--text)]"
            >
              {link.label}
            </a>
          </span>
        ))}
      </div>
    </footer>
  );
}
