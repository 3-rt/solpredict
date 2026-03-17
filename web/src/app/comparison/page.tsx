"use client";

import { useEffect, useState } from "react";
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

const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#1a1a2e",
    border: "1px solid #333",
    borderRadius: 8,
    color: "#e2e8f0",
    fontSize: 12,
  },
};

export default function ComparisonPage() {
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
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-red-950/40 border border-red-800 rounded-xl p-6 text-red-300 max-w-md text-center">
          <p className="font-semibold mb-1">Failed to load results</p>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[var(--text-muted)]">
        Loading results...
      </div>
    );
  }

  const rfMetrics = data.models.random_forest.test_metrics;
  const nnMetrics = data.models.neural_network.test_metrics;

  // Build scatter data
  const rfScatterData = data.plots.scatter.y_true.map((y, i) => ({
    actual: y,
    predicted: data.plots.scatter.rf_pred[i],
  }));
  const nnScatterData = data.plots.scatter.y_true.map((y, i) => ({
    actual: y,
    predicted: data.plots.scatter.nn_pred[i],
  }));

  // Bar chart data
  const barData = [
    { metric: "R²", RF: rfMetrics.r2, NN: nnMetrics.r2 },
    { metric: "RMSE", RF: rfMetrics.rmse, NN: nnMetrics.rmse },
    { metric: "MAE", RF: rfMetrics.mae, NN: nnMetrics.mae },
  ];

  return (
    <div className="min-h-screen px-6 py-10 max-w-5xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
          Model Comparison
        </h1>
        <p className="text-[var(--text-secondary)]">
          Side-by-side evaluation of Random Forest and Neural Network on the ESOL test set.
        </p>
      </div>

      {/* Metrics table */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-6 mb-8">
        <h2 className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
          Test Metrics
        </h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="text-left py-2 pr-4 text-[var(--text-muted)] font-medium">Metric</th>
              <th className="py-2 px-4 text-[var(--accent-blue)] font-semibold text-center">
                Random Forest
              </th>
              <th className="py-2 pl-4 text-[var(--accent-purple)] font-semibold text-center">
                Neural Network
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              { label: "R²", rf: rfMetrics.r2.toFixed(4), nn: nnMetrics.r2.toFixed(4), higherBetter: true },
              { label: "RMSE", rf: rfMetrics.rmse.toFixed(4), nn: nnMetrics.rmse.toFixed(4), higherBetter: false },
              { label: "MAE", rf: rfMetrics.mae.toFixed(4), nn: nnMetrics.mae.toFixed(4), higherBetter: false },
            ].map((row) => {
              const rfVal = parseFloat(row.rf);
              const nnVal = parseFloat(row.nn);
              const rfWins = row.higherBetter ? rfVal >= nnVal : rfVal <= nnVal;
              return (
                <tr key={row.label} className="border-b border-[var(--border)]/50">
                  <td className="py-3 pr-4 text-[var(--text-secondary)] font-medium">{row.label}</td>
                  <td className={`py-3 px-4 text-center font-mono ${rfWins ? "text-[var(--accent-blue)] font-bold" : "text-[var(--text-primary)]"}`}>
                    {row.rf}
                  </td>
                  <td className={`py-3 pl-4 text-center font-mono ${!rfWins ? "text-[var(--accent-purple)] font-bold" : "text-[var(--text-primary)]"}`}>
                    {row.nn}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Scatter plots */}
      <div className="grid grid-cols-2 gap-6 mb-8">
        {[
          { label: "Random Forest", color: "#60a5fa", scatterData: rfScatterData },
          { label: "Neural Network", color: "#a78bfa", scatterData: nnScatterData },
        ].map(({ label, color, scatterData }) => (
          <div key={label} className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color }}>
              {label} — Predicted vs Actual
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="actual"
                  name="Actual"
                  label={{ value: "Actual log(S)", position: "insideBottom", offset: -10, fill: "#94a3b8", fontSize: 11 }}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  domain={[-10, 2]}
                />
                <YAxis
                  dataKey="predicted"
                  name="Predicted"
                  label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 11 }}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  domain={[-10, 2]}
                />
                <Tooltip {...TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3" }} />
                <ReferenceLine
                  segment={[{ x: -10, y: -10 }, { x: 2, y: 2 }]}
                  stroke="#64748b"
                  strokeDasharray="4 4"
                />
                <Scatter data={scatterData} fill={color} fillOpacity={0.6} r={2} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      {/* Bar chart */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-6">
        <h2 className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
          Metrics Comparison
        </h2>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend
              wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
              formatter={(value) => (value === "RF" ? "Random Forest" : "Neural Network")}
            />
            <Bar dataKey="RF" fill="#60a5fa" radius={[4, 4, 0, 0]} />
            <Bar dataKey="NN" fill="#a78bfa" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
