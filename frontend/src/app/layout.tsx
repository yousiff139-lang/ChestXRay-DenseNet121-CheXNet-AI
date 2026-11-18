import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Provider from "../components/provider";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ChestXRay AI — Clinical Radiologist Workstation (DenseNet-121)",
  description:
    "Seminar 16: Chest X-ray disease classification system using pretrained DenseNet-121 and ResNet-50 baseline on NIH ChestX-ray14",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${inter.variable} antialiased min-h-screen font-sans bg-slate-950 text-slate-100 selection:bg-blue-600/30 selection:text-white`}
        suppressHydrationWarning
      >
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
