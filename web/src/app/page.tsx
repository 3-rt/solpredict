"use client";

import { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Legend,
  ReferenceLine,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Glucose", smiles: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
];

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

/* ─── Types ─── */

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

/* ─── Page ─── */

export default function Home() {
  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 pt-24 sm:pt-32 pb-12 sm:pb-16">
      <PredictSection />
      <Divider />
      <ComparisonSection />
      <Divider />
      <MethodologySection />
      <Divider />
      <Footer />
    </div>
  );
}

function Divider() {
  return <hr className="border-t border-[var(--border)] my-20" />;
}

function describeSolubility(logS: number): { label: string; description: string } {
  if (logS >= 0) return { label: "Highly soluble", description: "Dissolves very easily in water" };
  if (logS >= -1) return { label: "Soluble", description: "Dissolves readily in water" };
  if (logS >= -2) return { label: "Moderately soluble", description: "Dissolves with some difficulty" };
  if (logS >= -4) return { label: "Sparingly soluble", description: "Dissolves poorly in water" };
  if (logS >= -6) return { label: "Poorly soluble", description: "Very limited dissolution in water" };
  return { label: "Practically insoluble", description: "Essentially does not dissolve in water" };
}

/* ─── Predict ─── */

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
    handlePredict(mol.smiles);
  }

  return (
    <section>
      <h1 className="text-2xl font-bold mb-3">
        SolPredict
      </h1>
      <p className="text-[var(--text-secondary)] leading-relaxed mb-14">
        Predict aqueous solubility of organic molecules using Random Forest and Neural
        Network models trained on the{" "}
        <a
          href="https://pubs.acs.org/doi/10.1021/ci034243x"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-[var(--text)] transition-colors"
        >
          ESOL dataset
        </a>
        . Enter a SMILES string or try one of the examples.
      </p>

      {/* Input */}
      <div className="flex gap-2 mb-3">
        <input
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handlePredict()}
          placeholder="Enter SMILES (e.g. CCO)"
          className="flex-1 border border-[var(--border)] bg-white rounded-lg px-4 py-2.5 text-sm font-[family-name:var(--font-mono)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--teal)] transition-colors"
        />
        <button
          onClick={() => handlePredict()}
          disabled={loading || !smiles.trim()}
          className="bg-[var(--teal)] text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-40"
        >
          {loading ? "..." : "Predict"}
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5 mb-14">
        {EXAMPLES.map((mol) => (
          <button
            key={mol.name}
            onClick={() => handleExample(mol)}
            className="bg-[var(--bg-code)] rounded px-2.5 py-1 text-xs text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors"
          >
            {mol.name}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 rounded-lg p-3 mb-6 text-sm text-[var(--sienna)]">
          {error}
        </div>
      )}

      {/* Results */}
      {result && result.valid && (
        <div>
          {result.molecule_name && (
            <p className="text-[var(--text-secondary)] text-sm mb-5">
              Results for <span className="font-semibold text-[var(--text)]">{result.molecule_name}</span>
            </p>
          )}

          <div className="grid grid-cols-2 gap-4 mb-6">
            {[
              { label: "Random Forest", value: result.predictions?.random_forest, color: TEAL, tag: "sklearn" },
              { label: "Neural Network", value: result.predictions?.neural_network, color: SIENNA, tag: "PyTorch" },
            ].map((m) => {
              const interp = m.value != null ? describeSolubility(m.value) : null;
              return (
                <div
                  key={m.label}
                  className="bg-white border border-[var(--border)] rounded-lg p-5"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold" style={{ color: m.color }}>
                      {m.label}
                    </span>
                    <span className="text-[10px] text-[var(--text-muted)] font-[family-name:var(--font-mono)]">
                      {m.tag}
                    </span>
                  </div>
                  <div className="font-[family-name:var(--font-mono)] text-2xl font-medium mb-0.5">
                    {m.value?.toFixed(2) ?? "\u2014"}
                  </div>
                  <div className="text-xs text-[var(--text-muted)] mb-3">log mol/L</div>
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
            <div className="bg-white border border-[var(--border)] rounded-lg p-5">
              <div className="text-xs font-medium text-[var(--text-muted)] mb-3">
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
                    <div className="text-[10px] text-[var(--text-muted)] mt-0.5">{d.label}</div>
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

/* ─── Comparison ─── */

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
      <h2 className="text-xl font-bold mb-2">
        Model Comparison
      </h2>
      <p className="text-[var(--text-secondary)] text-sm mb-8">
        Side-by-side evaluation on the ESOL test set (20% holdout, n=226).
      </p>

      {/* Metrics table */}
      <div className="overflow-x-auto mb-10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="text-left py-2 pr-6 font-medium text-[var(--text-muted)]">Metric</th>
              <th className="py-2 px-4 font-semibold text-right" style={{ color: TEAL }}>Random Forest</th>
              <th className="py-2 pl-4 font-semibold text-right" style={{ color: SIENNA }}>Neural Network</th>
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
                  <td className={`py-3 px-4 text-right font-[family-name:var(--font-mono)] tabular-nums ${rfWins ? "font-semibold" : ""}`}
                    style={rfWins ? { color: TEAL } : undefined}
                  >
                    {row.rf.toFixed(4)}
                  </td>
                  <td className={`py-3 pl-4 text-right font-[family-name:var(--font-mono)] tabular-nums ${!rfWins ? "font-semibold" : ""}`}
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

      {/* Scatter plots */}
      <div className="grid grid-cols-2 gap-6 mb-10">
        {[
          { label: "Random Forest", color: TEAL, data: rfScatter },
          { label: "Neural Network", color: SIENNA, data: nnScatter },
        ].map(({ label, color, data: scatterData }) => (
          <div key={label}>
            <h3 className="text-xs font-semibold mb-3" style={{ color }}>
              {label} <span className="font-normal text-[var(--text-muted)]">— Predicted vs Actual</span>
            </h3>
            <div className="bg-white border border-[var(--border)] rounded-lg p-3">
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
                  <XAxis
                    dataKey="actual" name="Actual"
                    label={{ value: "Actual log(S)", position: "insideBottom", offset: -10, fill: "#999", fontSize: 10 }}
                    tick={{ fill: "#999", fontSize: 10 }} domain={[-10, 2]} stroke="#e5e2db"
                  />
                  <YAxis
                    dataKey="predicted" name="Predicted"
                    label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "#999", fontSize: 10 }}
                    tick={{ fill: "#999", fontSize: 10 }} domain={[-10, 2]} stroke="#e5e2db"
                  />
                  <Tooltip {...CHART_TOOLTIP} cursor={{ strokeDasharray: "3 3" }} />
                  <ReferenceLine segment={[{ x: -10, y: -10 }, { x: 2, y: 2 }]} stroke="#ccc" strokeDasharray="4 4" />
                  <Scatter data={scatterData} fill={color} fillOpacity={0.5} r={2.5} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>

      {/* Residuals */}
      <div className="grid grid-cols-2 gap-6 mb-10">
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
              <h3 className="text-xs font-semibold mb-3" style={{ color }}>
                {label} <span className="font-normal text-[var(--text-muted)]">— Residuals</span>
              </h3>
              <div className="bg-white border border-[var(--border)] rounded-lg p-3">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={bins} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
                    <XAxis dataKey="bin" tick={{ fill: "#999", fontSize: 9 }}
                      label={{ value: "Residual", position: "insideBottom", offset: -10, fill: "#999", fontSize: 10 }}
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

      {/* Bar chart */}
      <h3 className="text-xs font-semibold text-[var(--text-muted)] mb-3">Metrics Overview</h3>
      <div className="bg-white border border-[var(--border)] rounded-lg p-3">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={barData} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e2db" />
            <XAxis dataKey="metric" tick={{ fill: "#666", fontSize: 12 }} stroke="#e5e2db" />
            <YAxis tick={{ fill: "#999", fontSize: 11 }} stroke="#e5e2db" />
            <Tooltip {...CHART_TOOLTIP} />
            <Legend wrapperStyle={{ fontSize: 11 }}
              formatter={(v) => (v === "RF" ? "Random Forest" : "Neural Network")}
            />
            <Bar dataKey="RF" fill={TEAL} radius={[3, 3, 0, 0]} />
            <Bar dataKey="NN" fill={SIENNA} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

/* ─── Methodology ─── */

function MethodologySection() {
  return (
    <section>
      <h2 className="text-xl font-bold mb-2">
        Methodology
      </h2>
      <p className="text-[var(--text-secondary)] text-sm mb-8">
        How SolPredict works, from data to predictions.
      </p>

      {/* Dataset */}
      <h3 className="font-semibold text-base mb-2">Dataset</h3>
      <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-4">
        The{" "}
        <a href="https://pubs.acs.org/doi/10.1021/ci034243x" target="_blank" rel="noopener noreferrer"
          className="underline hover:text-[var(--text)] transition-colors">
          ESOL dataset
        </a>{" "}
        (Delaney, 2004) contains experimentally measured aqueous solubility for 1,128 small
        organic molecules. Target variable is log(S) in mol/L. We use an 80/20 train/test
        split with random seed 42.
      </p>

      {/* Features */}
      <h3 className="font-semibold text-base mb-2 mt-8">Features</h3>
      <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-4">
        Each molecule is converted to a 2048-bit Morgan fingerprint (ECFP4, radius=2) using
        RDKit. Each bit encodes the presence of a circular substructure, producing a
        fixed-length binary vector that standard ML algorithms can consume directly.
      </p>
      <div className="bg-[var(--bg-code)] rounded-lg p-4 font-[family-name:var(--font-mono)] text-xs leading-relaxed mb-4">
        <div className="text-[var(--text-muted)]"># RDKit pipeline</div>
        <div><span style={{ color: TEAL }}>from</span> rdkit <span style={{ color: TEAL }}>import</span> Chem</div>
        <div><span style={{ color: TEAL }}>from</span> rdkit.Chem <span style={{ color: TEAL }}>import</span> AllChem</div>
        <div className="mt-2">mol = Chem.MolFromSmiles(<span style={{ color: SIENNA }}>&quot;CC(=O)Oc1ccccc1C(=O)O&quot;</span>)</div>
        <div>fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=<span style={{ color: SIENNA }}>2</span>, nBits=<span style={{ color: SIENNA }}>2048</span>)</div>
      </div>

      {/* Models */}
      <h3 className="font-semibold text-base mb-3 mt-8">Models</h3>
      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <div className="text-xs font-semibold mb-2" style={{ color: TEAL }}>Random Forest</div>
          <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-3">
            Ensemble of 100 decision trees via bootstrap aggregation.
            Averages predictions to reduce variance.
          </p>
          <ul className="text-xs text-[var(--text-secondary)] space-y-1">
            <li><span className="text-[var(--text-muted)]">Library:</span> scikit-learn</li>
            <li><span className="text-[var(--text-muted)]">Input:</span> 2048-bit fingerprint</li>
          </ul>
        </div>
        <div>
          <div className="text-xs font-semibold mb-2" style={{ color: SIENNA }}>Neural Network</div>
          <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-3">
            Feed-forward MLP (2048 {"\u2192"} 512 {"\u2192"} 128 {"\u2192"} 1) with ReLU
            and dropout (p=0.2). Trained with Adam, MSE loss, 100 epochs.
          </p>
          <ul className="text-xs text-[var(--text-secondary)] space-y-1">
            <li><span className="text-[var(--text-muted)]">Library:</span> PyTorch</li>
            <li><span className="text-[var(--text-muted)]">LR:</span> 0.001, Batch: 64</li>
          </ul>
        </div>
      </div>

      {/* Evaluation */}
      <h3 className="font-semibold text-base mb-2 mt-8">Evaluation</h3>
      <p className="text-[var(--text-secondary)] text-sm leading-relaxed">
        Models are evaluated with <strong>R\u00B2</strong> (variance explained, higher is better),{" "}
        <strong>RMSE</strong> (root mean squared error, penalizes outliers), and{" "}
        <strong>MAE</strong> (mean absolute error, robust to outliers). All error metrics are
        in log mol/L units.
      </p>
    </section>
  );
}

/* ─── Footer ─── */

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
    <footer className="text-xs text-[var(--text-muted)] flex items-center justify-between">
      <span>Built by <a href="https://www.basilliu.dev/" target="_blank" rel="noopener noreferrer" className="underline hover:text-[var(--text)] transition-colors">Basil Liu</a></span>
      <div className="flex flex-wrap gap-3">
        {links.map((link, i) => (
          <span key={link.label}>
            {i > 0 && <span className="mr-3">{"\u00B7"}</span>}
            <a
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-[var(--text)] transition-colors"
            >
              {link.label}
            </a>
          </span>
        ))}
      </div>
    </footer>
  );
}
