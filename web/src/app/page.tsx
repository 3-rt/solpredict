"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Glucose", smiles: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
];

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

export default function PredictPage() {
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
      if (!data.valid) {
        setError(data.error || "Invalid SMILES string");
      }
    } catch {
      setError(
        "Could not reach prediction server. It may be waking up — try again in 30 seconds."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleExample(mol: { name: string; smiles: string }) {
    setSmiles(mol.smiles);
    handlePredict(mol.smiles);
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-12">
      {/* Hero */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-semibold mb-2">
          Molecular Solubility Prediction
        </h1>
        <p className="text-[var(--text-muted)] text-sm">
          Predict aqueous solubility using Random Forest and Neural Network
          models trained on the ESOL dataset
        </p>
      </div>

      {/* Input */}
      <div className="flex gap-2 mb-3">
        <input
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handlePredict()}
          placeholder="Enter SMILES string (e.g., CCO for ethanol)"
          className="flex-1 bg-[var(--bg-card)] border border-[var(--border)] rounded-lg px-4 py-3 text-sm font-mono text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-indigo)]"
        />
        <button
          onClick={() => handlePredict()}
          disabled={loading || !smiles.trim()}
          className="bg-[var(--accent-indigo)] text-white px-5 py-3 rounded-lg text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {/* Example pills */}
      <div className="flex flex-wrap gap-2 justify-center mb-10">
        {EXAMPLES.map((mol) => (
          <button
            key={mol.name}
            onClick={() => handleExample(mol)}
            className="bg-[var(--bg-card)] border border-[var(--border)] rounded-full px-3 py-1 text-xs text-[var(--text-secondary)] hover:border-[var(--accent-indigo)] transition-colors"
          >
            {mol.name}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-6 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      {result && result.valid && (
        <>
          {/* Molecule name */}
          {result.molecule_name && (
            <p className="text-center text-[var(--text-secondary)] text-sm mb-4">
              Showing predictions for{" "}
              <span className="font-semibold text-[var(--text-primary)]">
                {result.molecule_name}
              </span>
            </p>
          )}

          {/* Prediction cards */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            {/* Random Forest */}
            <div className="bg-[var(--bg-secondary)] border border-blue-900/50 rounded-xl p-5">
              <div className="flex justify-between items-center mb-3">
                <span className="text-xs text-[var(--accent-blue)] uppercase tracking-wider font-semibold">
                  Random Forest
                </span>
                <span className="text-[10px] text-[var(--text-muted)] bg-[#1e293b] px-2 py-0.5 rounded">
                  sklearn
                </span>
              </div>
              <div className="text-3xl font-bold mb-1">
                {result.predictions?.random_forest?.toFixed(2) ?? "—"}
              </div>
              <div className="text-xs text-[var(--text-muted)]">log mol/L</div>
            </div>

            {/* Neural Network */}
            <div className="bg-[var(--bg-secondary)] border border-purple-900/50 rounded-xl p-5">
              <div className="flex justify-between items-center mb-3">
                <span className="text-xs text-[var(--accent-purple)] uppercase tracking-wider font-semibold">
                  Neural Network
                </span>
                <span className="text-[10px] text-[var(--text-muted)] bg-[#1e293b] px-2 py-0.5 rounded">
                  PyTorch
                </span>
              </div>
              <div className="text-3xl font-bold mb-1">
                {result.predictions?.neural_network?.toFixed(2) ?? "—"}
              </div>
              <div className="text-xs text-[var(--text-muted)]">log mol/L</div>
            </div>
          </div>

          {/* Molecular properties */}
          {result.descriptors && (
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-5">
              <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-3">
                Molecular Properties
              </div>
              <div className="grid grid-cols-5 gap-4 text-center">
                <div>
                  <div className="text-lg font-semibold">
                    {result.descriptors.molecular_weight}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">
                    Mol. Weight
                  </div>
                </div>
                <div>
                  <div className="text-lg font-semibold">
                    {result.descriptors.logp}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">
                    LogP
                  </div>
                </div>
                <div>
                  <div className="text-lg font-semibold">
                    {result.descriptors.hbd}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">
                    H-Bond Donors
                  </div>
                </div>
                <div>
                  <div className="text-lg font-semibold">
                    {result.descriptors.hba}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">
                    H-Bond Acceptors
                  </div>
                </div>
                <div>
                  <div className="text-lg font-semibold">
                    {result.descriptors.tpsa}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">
                    TPSA
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
