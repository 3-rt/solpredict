export default function MethodologyPage() {
  return (
    <div className="min-h-screen px-6 py-10 max-w-4xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">Methodology</h1>
        <p className="text-[var(--text-secondary)]">
          An overview of the data, features, and models used in SolPredict.
        </p>
      </div>

      {/* Section 1: Dataset */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">
          1. Dataset: ESOL
        </h2>
        <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-6">
          The ESOL (Estimated SOLubility) dataset, published by John Delaney in 2004, contains
          experimentally measured aqueous solubility values for 1,128 small organic molecules.
          The target variable is log(S), where S is solubility in mol/L. It is one of the
          most widely used benchmark datasets in molecular property prediction.
        </p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Total Molecules", value: "1,128" },
            { label: "Train / Test Split", value: "80 / 20" },
            { label: "Random Seed", value: "42" },
          ].map((stat) => (
            <div
              key={stat.label}
              className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 text-center"
            >
              <div className="text-2xl font-bold text-[var(--accent-indigo)] mb-1">{stat.value}</div>
              <div className="text-xs text-[var(--text-secondary)]">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Section 2: Feature Engineering */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">
          2. Feature Engineering: Morgan Fingerprints
        </h2>
        <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-4">
          Each molecule is represented as a 2048-bit Morgan fingerprint (also known as ECFP4 —
          Extended Connectivity Fingerprints with diameter 4). Morgan fingerprints are circular
          fingerprints that encode the local chemical environment of each atom up to a given
          radius. At radius=2, each atom considers neighbors within 2 bonds, capturing
          functional group context effectively.
        </p>
        <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-4">
          Each bit in the resulting 2048-dimensional vector indicates the presence or absence of
          a particular circular substructure. This fixed-length binary representation allows
          standard machine learning algorithms to operate on molecular data without requiring
          graph neural networks.
        </p>
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-5 font-mono text-sm">
          <div className="text-[var(--text-muted)] mb-2 text-xs"># Python / RDKit pipeline</div>
          <div className="text-[var(--accent-blue)]">from rdkit <span className="text-[var(--text-primary)]">import</span> Chem</div>
          <div className="text-[var(--accent-blue)]">from rdkit.Chem <span className="text-[var(--text-primary)]">import</span> AllChem</div>
          <div className="mt-2 text-[var(--text-muted)]"># SMILES string → RDKit molecule</div>
          <div>
            <span className="text-[var(--accent-indigo)]">mol</span>
            <span className="text-[var(--text-primary)]"> = Chem.MolFromSmiles(</span>
            <span className="text-green-400">&quot;CC(=O)Oc1ccccc1C(=O)O&quot;</span>
            <span className="text-[var(--text-primary)]">)</span>
          </div>
          <div className="mt-1 text-[var(--text-muted)]"># Molecule → Morgan fingerprint (radius=2, 2048 bits)</div>
          <div>
            <span className="text-[var(--accent-indigo)]">fp</span>
            <span className="text-[var(--text-primary)]"> = AllChem.GetMorganFingerprintAsBitVect(</span>
          </div>
          <div className="pl-8 text-[var(--text-primary)]">
            mol, <span className="text-[var(--accent-purple)]">radius</span>=<span className="text-orange-400">2</span>,
            <span className="text-[var(--accent-purple)]"> nBits</span>=<span className="text-orange-400">2048</span>
          </div>
          <div className="text-[var(--text-primary)]">)</div>
          <div className="mt-1 text-[var(--text-muted)]"># Result: array of 2048 bits → model input</div>
          <div>
            <span className="text-[var(--accent-indigo)]">X</span>
            <span className="text-[var(--text-primary)]"> = np.array(fp)  </span>
            <span className="text-[var(--text-muted)]"># shape: (2048,)</span>
          </div>
        </div>
      </section>

      {/* Section 3: Models */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">
          3. Models
        </h2>
        <div className="grid grid-cols-2 gap-6">
          {/* Random Forest */}
          <div className="bg-[var(--bg-card)] border border-[var(--accent-blue)]/30 rounded-xl p-6">
            <div className="text-xs font-semibold text-[var(--accent-blue)] uppercase tracking-wider mb-3">
              Random Forest
            </div>
            <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-4">
              An ensemble of 100 decision trees trained via bootstrap aggregation (bagging).
              Each tree is trained on a random subset of samples and features. The final
              prediction is the average across all trees, reducing variance and improving
              generalization.
            </p>
            <ul className="text-sm text-[var(--text-secondary)] space-y-1">
              <li><span className="text-[var(--text-muted)]">Library:</span> scikit-learn</li>
              <li><span className="text-[var(--text-muted)]">n_estimators:</span> 100</li>
              <li><span className="text-[var(--text-muted)]">Input:</span> 2048-bit fingerprint</li>
              <li><span className="text-[var(--text-muted)]">Output:</span> log(S) regression</li>
            </ul>
          </div>

          {/* Neural Network */}
          <div className="bg-[var(--bg-card)] border border-[var(--accent-purple)]/30 rounded-xl p-6">
            <div className="text-xs font-semibold text-[var(--accent-purple)] uppercase tracking-wider mb-3">
              Neural Network
            </div>
            <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-4">
              A fully-connected feed-forward network with dropout regularization, trained
              with Adam optimizer and MSE loss. Batch normalization is applied after each
              hidden layer. Dropout (p=0.2) prevents overfitting.
            </p>
            <ul className="text-sm text-[var(--text-secondary)] space-y-1">
              <li><span className="text-[var(--text-muted)]">Library:</span> PyTorch</li>
              <li><span className="text-[var(--text-muted)]">Architecture:</span> 2048 → 512 → 128 → 1</li>
              <li><span className="text-[var(--text-muted)]">Epochs:</span> 100, LR: 0.001, Batch: 64</li>
              <li><span className="text-[var(--text-muted)]">Dropout:</span> 0.2</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Section 4: Evaluation */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">
          4. Evaluation Metrics
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {[
            {
              label: "R² (Coefficient of Determination)",
              accent: "var(--accent-indigo)",
              desc: "Measures the proportion of variance in the target explained by the model. A perfect model scores R²=1. Values closer to 1 indicate a better fit. R² can be negative if the model performs worse than a simple mean baseline.",
            },
            {
              label: "RMSE (Root Mean Squared Error)",
              accent: "var(--accent-blue)",
              desc: "The square root of the average squared differences between predicted and actual values. RMSE penalizes large errors more heavily than MAE, making it sensitive to outliers. Reported in the same units as the target (log mol/L).",
            },
            {
              label: "MAE (Mean Absolute Error)",
              accent: "var(--accent-purple)",
              desc: "The average absolute difference between predictions and true values. More robust to outliers than RMSE. Directly interpretable: an MAE of 0.88 means predictions are off by 0.88 log units on average.",
            },
          ].map((m) => (
            <div
              key={m.label}
              className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5"
            >
              <div
                className="text-sm font-semibold mb-2"
                style={{ color: m.accent }}
              >
                {m.label}
              </div>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
