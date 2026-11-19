"use client";

import React, { useState } from "react";
import { DiseaseClassification } from "../lib/types";
import { AlertCircle, CheckCircle2, HelpCircle } from "lucide-react";

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
    <div className="bg-slate-900/70 rounded-xl border border-slate-800 p-3.5 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 pb-2.5 border-b border-slate-800 mb-2.5">
        <div>
          <h4 className="text-xs font-semibold text-slate-100">
            Pathology Probabilities
          </h4>
          <p className="text-[10px] text-slate-400">
            DenseNet-121 multi-label inference
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
            All (14)
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
      <div className="space-y-1.5 overflow-y-auto pr-1 max-h-[300px] custom-scrollbar">
        {displayed.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs">
            No abnormal findings detected.
          </div>
        ) : (
          displayed.map((item) => {
            const percentage = (item.probability * 100).toFixed(1);
            const state = item.state || (item.is_positive ? "Positive" : "Negative");
            const isPositive = state === "Positive";
            const isUncertain = state === "Uncertain";
            const threshold = item.threshold !== undefined ? item.threshold : confidenceThreshold;

            return (
              <div
                key={item.disease_name}
                className={`p-2 rounded-lg border text-xs transition ${
                  isPositive
                    ? "bg-rose-950/20 border-rose-500/30"
                    : isUncertain
                    ? "bg-amber-950/20 border-amber-500/30"
                    : "bg-slate-950/50 border-slate-800/80"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    {isPositive ? (
                      <AlertCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    ) : isUncertain ? (
                      <HelpCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-3.5 h-3.5 text-slate-600 shrink-0" />
                    )}
                    <span
                      className={`font-medium ${
                        isPositive
                          ? "text-rose-200 font-semibold"
                          : isUncertain
                          ? "text-amber-200 font-semibold"
                          : "text-slate-300"
                      }`}
                    >
                      {item.disease_name.replace("_", " ")}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[9px] font-semibold px-1.5 py-0.5 rounded capitalize ${
                        isPositive
                          ? "bg-rose-500/20 text-rose-300"
                          : isUncertain
                          ? "bg-amber-500/20 text-amber-300"
                          : "text-slate-500"
                      }`}
                    >
                      {state}
                    </span>

                    <span
                      className={`font-mono text-xs font-semibold ${
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

                {/* Progress bar */}
                <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden relative">
                  <div
                    className="absolute top-0 bottom-0 w-px bg-slate-500 z-10"
                    style={{ left: `${Math.min(threshold * 100, 100)}%` }}
                    title={`Threshold: ${(threshold * 100).toFixed(0)}%`}
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
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
