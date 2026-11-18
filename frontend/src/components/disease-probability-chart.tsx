"use client";

import React, { useState } from "react";
import { DiseaseClassification } from "../lib/types";
import { DISEASE_INFO } from "../lib/constants";
import { AlertCircle, CheckCircle2, HelpCircle, ShieldAlert } from "lucide-react";

interface DiseaseProbabilityChartProps {
  predictions: DiseaseClassification[];
  confidenceThreshold?: number;
}

export const DiseaseProbabilityChart: React.FC<DiseaseProbabilityChartProps> = ({
  predictions,
  confidenceThreshold = 0.5,
}) => {
  const [filter, setFilter] = useState<"all" | "abnormal">("all");

  const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
  const positiveCount = sorted.filter((p) => p.is_positive || p.state === "Positive").length;
  const uncertainCount = sorted.filter((p) => p.state === "Uncertain").length;
  const displayed =
    filter === "abnormal"
      ? sorted.filter((p) => p.is_positive || p.state === "Positive" || p.state === "Uncertain")
      : sorted;

  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800 p-4 flex flex-col h-full">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800 mb-3">
        <div>
          <h4 className="text-sm font-semibold text-slate-100">
            Multi-Label Pathology Spectrum
          </h4>
          <p className="text-xs text-slate-400">
            Calibrated sigmoid probabilities &amp; validation thresholds (Youden&apos;s J)
          </p>
        </div>

        {/* Filter toggle */}
        <div className="flex items-center gap-1 bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-[11px]">
          <button
            type="button"
            onClick={() => setFilter("all")}
            className={`px-2 py-0.5 rounded font-medium transition ${
              filter === "all"
                ? "bg-slate-800 text-slate-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            All 14 Classes
          </button>
          <button
            type="button"
            onClick={() => setFilter("abnormal")}
            className={`px-2 py-0.5 rounded font-medium transition flex items-center gap-1 ${
              filter === "abnormal"
                ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Abnormal ({positiveCount + uncertainCount})
          </button>
        </div>
      </div>

      {/* Disease List */}
      <div className="space-y-2 overflow-y-auto pr-1 max-h-[460px] custom-scrollbar flex-1">
        {displayed.length === 0 ? (
          <div className="py-12 text-center text-slate-500 text-xs">
            No abnormalities or uncertain findings detected.
          </div>
        ) : (
          displayed.map((item) => {
            const percentage = (item.probability * 100).toFixed(1);
            const state = item.state || (item.is_positive ? "Positive" : "Negative");
            const isPositive = state === "Positive";
            const isUncertain = state === "Uncertain";
            const threshold = item.threshold !== undefined ? item.threshold : confidenceThreshold;
            const thresholdPct = (threshold * 100).toFixed(1);
            const prevalencePct = item.prevalence !== undefined && item.prevalence !== null
              ? (item.prevalence * 100).toFixed(1)
              : null;
            const info = DISEASE_INFO[item.disease_name];

            return (
              <div
                key={item.disease_name}
                className={`p-2.5 rounded-lg border transition-colors ${
                  isPositive
                    ? "bg-rose-950/20 border-rose-500/35"
                    : isUncertain
                    ? "bg-amber-950/20 border-amber-500/35"
                    : "bg-slate-950/40 border-slate-800/80"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    {isPositive ? (
                      <AlertCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    ) : isUncertain ? (
                      <HelpCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-3.5 h-3.5 text-slate-600 shrink-0" />
                    )}
                    <span
                      className={`text-xs ${
                        isPositive
                          ? "font-semibold text-rose-200"
                          : isUncertain
                          ? "font-semibold text-amber-200"
                          : "font-medium text-slate-300"
                      }`}
                    >
                      {item.disease_name.replace("_", " ")}
                    </span>
                    {info && (
                      <span className="text-[10px] font-mono text-slate-400 border border-slate-700/60 rounded px-1.5 py-0.2">
                        {info.clinicalConcern}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* 3-State Badge */}
                    <span
                      className={`text-[9px] uppercase font-mono font-bold px-1.5 py-0.5 rounded border ${
                        isPositive
                          ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                          : isUncertain
                          ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                          : "bg-slate-800 text-slate-400 border-slate-700"
                      }`}
                    >
                      {state}
                    </span>

                    <span
                      className={`text-xs font-mono font-bold ${
                        isPositive
                          ? "text-rose-400"
                          : isUncertain
                          ? "text-amber-400"
                          : "text-slate-400"
                      }`}
                    >
                      {percentage}%
                    </span>
                  </div>
                </div>

                {/* Progress bar with per-class threshold marker */}
                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden relative">
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-slate-400 z-10"
                    style={{ left: `${Math.min(threshold * 100, 100)}%` }}
                    title={`Validation threshold: ${thresholdPct}%`}
                  />
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      isPositive
                        ? "bg-rose-500"
                        : isUncertain
                        ? "bg-amber-500"
                        : "bg-slate-600"
                    }`}
                    style={{ width: `${Math.min(item.probability * 100, 100)}%` }}
                  />
                </div>

                {/* Sub-label: Cutoff and Prevalence info */}
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono mt-1">
                  <span>Cutoff: {thresholdPct}%</span>
                  {prevalencePct && <span>Prevalence: {prevalencePct}%</span>}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Mandatory Disclaimer Footer */}
      <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center gap-1.5 text-[11px] text-slate-400">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-500 shrink-0" />
        <span>Research prototype. Not for clinical use.</span>
      </div>
    </div>
  );
};
