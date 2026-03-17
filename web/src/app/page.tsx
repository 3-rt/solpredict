"use client";

import { useState } from "react";

const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Glucose", smiles: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
];

interface PredictResult {
  rf_prediction: number;
  nn_prediction: number;
  molecular_properties: {
    molecular_weight: number;
    logp: number;
    hbd: number;
    hba: number;
    tpsa: number;
  };
}

export default function PredictPage() {
  const [smiles, setSmiles] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    if (!smiles.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: smiles.trim() }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data?.detail ?? `Server error: ${res.status}`);
      }

      const data: PredictResult = await res.json();
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setError("Cannot reach the prediction API. Make sure the backend is running.");
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen px-6 py-10 max-w-3xl mx-auto">
      {/* Header */}
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
          Predict Solubility
        </h1>
        <p className="text-[var(--text-secondary)]">
          Enter a SMILES string to predict aqueous solubility (log mol/L) using
          Random Forest and Neural Network models.
        </p>
      </div>

      {/* Input */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-6 mb-6">
        <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
          SMILES String
        </label>
        <div className="flex gap-3">
          <input
            type="text"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handlePredict()}
            placeholder="e.g. CCO"
            className="flex-1 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-4 py-2.5 text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-indigo)] transition-colors text-sm font-mono"
          />
          <button
            onClick={handlePredict}
            disabled={loading || !smiles.trim()}
            className="px-6 py-2.5 rounded-lg font-semibold text-sm bg-[var(--accent-indigo)] text-white hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {/* Example pills */}
        <div className="mt-4">
          <p className="text-xs text-[var(--text-muted)] mb-2">Examples:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.name}
                onClick={() => {
                  setSmiles(ex.smiles);
                  setResult(null);
                  setError(null);
                }}
                className="px-3 py-1 text-xs rounded-full border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent-indigo)] hover:text-[var(--accent-indigo)] transition-colors"
              >
                {ex.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-950/40 border border-red-800 rounded-xl p-4 mb-6 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Prediction cards */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            {/* RF card */}
            <div className="bg-[var(--bg-card)] border border-[var(--accent-blue)]/40 rounded-xl p-6">
              <div className="text-xs font-semibold text-[var(--accent-blue)] uppercase tracking-wider mb-1">
                Random Forest
              </div>
              <div className="text-4xl font-bold text-[var(--accent-blue)] mb-1">
                {result.rf_prediction.toFixed(3)}
              </div>
              <div className="text-xs text-[var(--text-muted)]">log(S) mol/L</div>
            </div>

            {/* NN card */}
            <div className="bg-[var(--bg-card)] border border-[var(--accent-purple)]/40 rounded-xl p-6">
              <div className="text-xs font-semibold text-[var(--accent-purple)] uppercase tracking-wider mb-1">
                Neural Network
              </div>
              <div className="text-4xl font-bold text-[var(--accent-purple)] mb-1">
                {result.nn_prediction.toFixed(3)}
              </div>
              <div className="text-xs text-[var(--text-muted)]">log(S) mol/L</div>
            </div>
          </div>

          {/* Molecular properties */}
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-6">
            <h3 className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-4">
              Molecular Properties
            </h3>
            <div className="grid grid-cols-5 gap-4">
              {[
                { label: "Mol. Weight", value: result.molecular_properties.molecular_weight.toFixed(1), unit: "g/mol" },
                { label: "LogP", value: result.molecular_properties.logp.toFixed(2), unit: "" },
                { label: "HBD", value: result.molecular_properties.hbd.toString(), unit: "donors" },
                { label: "HBA", value: result.molecular_properties.hba.toString(), unit: "acceptors" },
                { label: "TPSA", value: result.molecular_properties.tpsa.toFixed(1), unit: "Å²" },
              ].map((prop) => (
                <div key={prop.label} className="text-center">
                  <div className="text-lg font-semibold text-[var(--text-primary)]">{prop.value}</div>
                  {prop.unit && <div className="text-xs text-[var(--text-muted)]">{prop.unit}</div>}
                  <div className="text-xs text-[var(--text-secondary)] mt-1">{prop.label}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
