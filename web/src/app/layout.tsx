import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SolPredict — Molecular Solubility Prediction",
  description:
    "Predict aqueous solubility using Random Forest and Neural Network models trained on the ESOL dataset.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={mono.variable}>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
