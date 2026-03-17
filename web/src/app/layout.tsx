import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "SolPredict — Molecular Solubility Prediction",
  description:
    "Predict aqueous solubility using Random Forest and Neural Network models trained on the ESOL dataset.",
};

function Navbar() {
  const links = [
    { href: "/", label: "Predict" },
    { href: "/comparison", label: "Model Comparison" },
    { href: "/methodology", label: "Methodology" },
    { href: "/about", label: "About" },
  ];

  return (
    <nav className="flex items-center justify-between px-6 py-3 border-b border-[var(--border)] bg-[#0f0f18]">
      <Link href="/" className="flex items-center gap-2 font-bold text-[var(--text-primary)]">
        <span className="text-lg">⚗️</span>
        <span>SolPredict</span>
      </Link>
      <div className="flex items-center gap-6 text-sm text-[var(--text-secondary)]">
        {links.map((link) => (
          <Link key={link.href} href={link.href} className="hover:text-[var(--accent-indigo)] transition-colors">
            {link.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
