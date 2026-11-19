"use client";

import dynamic from "next/dynamic";

const ChestDashboard = dynamic(
  () => import("../components/chest-dashboard").then((mod) => mod.ChestDashboard),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen w-screen bg-slate-950 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-slate-400">
          <div className="w-8 h-8 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
          <span className="text-xs font-medium tracking-wide">
            Loading Diagnostic Workstation...
          </span>
        </div>
      </div>
    ),
  }
);

export default function Home() {
  return <ChestDashboard />;
}
