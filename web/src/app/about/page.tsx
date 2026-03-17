export default function AboutPage() {
  const mlStack = [
    { name: "RDKit", desc: "Molecular fingerprint computation and cheminformatics" },
    { name: "scikit-learn", desc: "Random Forest regression and preprocessing" },
    { name: "PyTorch", desc: "Neural network architecture and training" },
    { name: "pandas", desc: "Data loading, cleaning, and manipulation" },
  ];

  const webStack = [
    { name: "Next.js 15", desc: "React framework with App Router and SSR" },
    { name: "Tailwind CSS v4", desc: "Utility-first styling with CSS variables" },
    { name: "Recharts", desc: "Declarative chart components for React" },
    { name: "FastAPI", desc: "High-performance Python API backend" },
  ];

  return (
    <div className="min-h-screen px-6 py-10 max-w-3xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">About</h1>
        <p className="text-[var(--text-secondary)]">
          SolPredict — a molecular solubility prediction web application.
        </p>
      </div>

      {/* Project Motivation */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">
          Project Motivation
        </h2>
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-6">
          <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-3">
            Aqueous solubility is one of the most important physicochemical properties in
            drug discovery. A promising drug candidate with poor solubility will have
            inadequate bioavailability, making it difficult or impossible to administer
            effectively. Experimentally measuring solubility for thousands of candidate
            molecules is expensive and time-consuming.
          </p>
          <p className="text-[var(--text-secondary)] text-sm leading-relaxed mb-3">
            Machine learning offers a way to screen molecules computationally before synthesis.
            By training on the well-established ESOL benchmark dataset, SolPredict demonstrates
            that Morgan fingerprints combined with standard ML models can predict log-scale
            solubility with reasonable accuracy — providing a fast, accessible screening tool.
          </p>
          <p className="text-[var(--text-secondary)] text-sm leading-relaxed">
            This project also serves as an end-to-end template for deploying cheminformatics
            ML models as interactive web applications, bridging the gap between research
            notebooks and usable tools.
          </p>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">
          Tech Stack
        </h2>
        <div className="grid grid-cols-2 gap-6">
          {/* ML Pipeline */}
          <div>
            <div className="text-xs font-semibold text-[var(--accent-blue)] uppercase tracking-wider mb-3">
              ML Pipeline
            </div>
            <div className="space-y-3">
              {mlStack.map((item) => (
                <div
                  key={item.name}
                  className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-4"
                >
                  <div className="text-sm font-semibold text-[var(--text-primary)] mb-1">
                    {item.name}
                  </div>
                  <div className="text-xs text-[var(--text-secondary)]">{item.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Web Stack */}
          <div>
            <div className="text-xs font-semibold text-[var(--accent-purple)] uppercase tracking-wider mb-3">
              Web Stack
            </div>
            <div className="space-y-3">
              {webStack.map((item) => (
                <div
                  key={item.name}
                  className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-4"
                >
                  <div className="text-sm font-semibold text-[var(--text-primary)] mb-1">
                    {item.name}
                  </div>
                  <div className="text-xs text-[var(--text-secondary)]">{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Links */}
      <section>
        <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-4">Links</h2>
        <div className="flex flex-col gap-3">
          {[
            {
              label: "GitHub Repository",
              href: "https://github.com/example/solpredict",
              desc: "Source code for the ML pipeline and web application",
              accent: "var(--accent-indigo)",
            },
            {
              label: "Training Notebook",
              href: "https://github.com/example/solpredict/blob/main/notebook.ipynb",
              desc: "Jupyter notebook with exploratory data analysis and model training",
              accent: "var(--accent-blue)",
            },
            {
              label: "ESOL Dataset Paper",
              href: "https://pubs.acs.org/doi/10.1021/ci034243x",
              desc: "Delaney, J.S. (2004) — original ESOL publication in J. Chem. Inf. Comput. Sci.",
              accent: "var(--accent-purple)",
            },
          ].map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 flex items-start gap-4 hover:border-[var(--accent-indigo)]/50 transition-colors group"
            >
              <div
                className="mt-0.5 w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: link.accent }}
              />
              <div>
                <div
                  className="text-sm font-semibold group-hover:opacity-90 transition-opacity"
                  style={{ color: link.accent }}
                >
                  {link.label}
                </div>
                <div className="text-xs text-[var(--text-secondary)] mt-0.5">{link.desc}</div>
              </div>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
