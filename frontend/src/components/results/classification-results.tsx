"use client";

import React, { useState } from "react";
import {
  ClassificationResponse,
  DiagnosticReport,
  DicomMetadata,
} from "../../lib/types";
import { DISEASE_INFO } from "../../lib/constants";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  Clock,
  Info,
  ShieldAlert,
  Copy,
  Check,
  Stethoscope,
} from "lucide-react";

interface ClassificationResultsProps {
  result: ClassificationResponse;
  metadata?: DicomMetadata | null;
  onGenerateReport?: () => void;
  report?: DiagnosticReport | null;
  generatingReport?: boolean;
}

export const ClassificationResults: React.FC<ClassificationResultsProps> = ({
  result,
  metadata,
  onGenerateReport,
  report,
  generatingReport = false,
}) => {
  const [showDicom, setShowDicom] = useState(false);
  const [copied, setCopied] = useState(false);
  const hasPositive = result.positive_findings.some(
    (f) => f.is_positive || f.state === "Positive"
  );
  const hasFindings = result.positive_findings.length > 0;

  const copyReportText = () => {
    if (!report) return;
    const text = `CHEST RADIOGRAPHY AI EVALUATION REPORT
Model: ${result.model_used}
Latency: ${result.inference_time_ms.toFixed(1)} ms
Decision Threshold: ${(result.confidence_threshold || 0.5) * 100}%
Urgency: ${report.urgency_level.toUpperCase()}

CLINICAL SUMMARY:
${report.clinical_summary}

IMPRESSION:
${report.impression}

RECOMMENDATIONS:
${report.recommendations?.map((r, i) => `${i + 1}. ${r}`).join("\n")}

DISCLAIMER:
${report.disclaimer || result.disclaimer || "Research prototype. Not for clinical use."}
`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      {/* Top Banner: Status + Latency */}
      <div className="bg-slate-900/60 rounded-xl border border-slate-800 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            {hasPositive ? (
              <div className="w-8 h-8 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400">
                <AlertTriangle className="w-4 h-4" />
              </div>
            ) : hasFindings ? (
              <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <AlertTriangle className="w-4 h-4" />
              </div>
            ) : (
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
              </div>
            )}
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                {hasPositive
                  ? `${result.positive_findings.filter((f) => f.state === "Positive" || f.is_positive).length} Thoracic Abnormalities Detected`
                  : hasFindings
                  ? `${result.positive_findings.length} Borderline / Uncertain Findings`
                  : "No Acute Thoracic Abnormalities Detected"}
              </h3>
              <p className="text-xs text-slate-400">
                Model: <span className="text-slate-200 font-medium">{result.model_used}</span> |
                Validation Decision Thresholds (Youden&apos;s J)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="px-2.5 py-1 rounded bg-slate-950 border border-slate-800 text-xs text-slate-400 flex items-center gap-1.5 font-mono text-[11px]">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              <span>
                {result.model_inference_ms
                  ? `Model: ${result.model_inference_ms.toFixed(1)}ms | E2E: ${(result.end_to_end_ms || result.inference_time_ms).toFixed(1)}ms`
                  : `${result.inference_time_ms.toFixed(1)} ms`}
              </span>
            </div>

            {onGenerateReport && (
              <button
                type="button"
                onClick={onGenerateReport}
                disabled={generatingReport}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition disabled:opacity-50"
              >
                <Stethoscope className="w-3.5 h-3.5" />
                {generatingReport ? "Synthesizing Report..." : "Generate Clinical Impression"}
              </button>
            )}
          </div>
        </div>

        {/* Positive & Uncertain findings cards */}
        {hasFindings ? (
          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
            {result.positive_findings.map((finding) => {
              const info = DISEASE_INFO[finding.disease_name];
              const pct = (finding.probability * 100).toFixed(1);
              const state = finding.state || (finding.is_positive ? "Positive" : "Uncertain");
              const isPos = state === "Positive";

              return (
                <div
                  key={finding.disease_name}
                  className={`p-3 rounded-lg border flex flex-col justify-between ${
                    isPos
                      ? "bg-slate-950/60 border-rose-500/30"
                      : "bg-slate-950/60 border-amber-500/30"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <span className={`text-xs font-semibold ${isPos ? "text-rose-300" : "text-amber-300"}`}>
                        {finding.disease_name.replace("_", " ")}
                      </span>
                      <span
                        className={`text-[9px] uppercase font-mono font-bold px-1.5 py-0.2 rounded border ${
                          isPos
                            ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                            : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                        }`}
                      >
                        {state}
                      </span>
                    </div>
                    <span className={`text-xs font-mono font-bold ${isPos ? "text-rose-400" : "text-amber-400"}`}>
                      {pct}%
                    </span>
                  </div>
                  {info && (
                    <p className="text-[11px] text-slate-400">
                      {info.description}
                    </p>
                  )}
                  {finding.threshold !== undefined && (
                    <div className="mt-2 text-[10px] text-slate-500 font-mono flex items-center justify-between">
                      <span>Cutoff: {(finding.threshold * 100).toFixed(1)}%</span>
                      {finding.prevalence !== undefined && finding.prevalence !== null && (
                        <span>Prevalence: {(finding.prevalence * 100).toFixed(1)}%</span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="mt-3 p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-emerald-300 text-xs">
            Cardiothoracic ratio, pulmonary parenchyma, pleural spaces, and mediastinum are within normal radiological limits based on model classification.
          </div>
        )}

        {/* Research Disclaimer */}
        <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center gap-1.5 text-[11px] text-slate-400">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-500 shrink-0" />
          <span>{result.disclaimer || "Research prototype. Not for clinical use."}</span>
        </div>

        {/* DICOM metadata toggle */}
        {metadata && (
          <div className="mt-3 pt-3 border-t border-slate-800">
            <button
              type="button"
              onClick={() => setShowDicom(!showDicom)}
              className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 transition"
            >
              <Info className="w-3.5 h-3.5" />
              {showDicom ? "Hide DICOM Header Details" : "Show DICOM Header Details"}
            </button>

            {showDicom && (
              <div className="mt-2.5 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs p-3 rounded-lg bg-slate-950 border border-slate-800">
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-mono">Patient ID:</span>
                  <span className="text-slate-300 font-mono text-[11px]">{metadata.patient_id || "ANONYMIZED"}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-mono">Sex / Age:</span>
                  <span className="text-slate-300 font-mono text-[11px]">
                    {metadata.patient_sex || "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-mono">Modality:</span>
                  <span className="text-slate-300 font-mono text-[11px]">{metadata.modality || "CR/DX"}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-mono">Dimensions:</span>
                  <span className="text-slate-300 font-mono text-[11px]">
                    {metadata.columns || 0} × {metadata.rows || 0}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* AI Diagnostic Report Section (if generated) */}
      {report && (() => {
        const reportAny = report as unknown as {
          report?: string;
          summary?: string;
          severity_level?: string;
          clinical_summary?: string;
          impression?: string;
          urgency_level?: string;
          recommendations?: string[];
          disclaimer?: string;
        };
        const fullMarkdown = reportAny?.report;
        const summary = report.clinical_summary || reportAny?.summary;
        const impression = report.impression;
        const recommendations = report.recommendations || [];
        const urgency = (report.urgency_level || reportAny?.severity_level || "standard").toUpperCase();
        const disclaimer = report.disclaimer || "AI-assisted preliminary interpretation for research/educational use. Requires clinical review by a certified radiologist.";

        return (
          <div className="bg-slate-900/60 rounded-xl border border-slate-800 p-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-semibold text-slate-100">
                  Structured Clinical Diagnostic Report
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`text-[10px] font-mono font-semibold uppercase px-2 py-0.5 rounded border ${
                    urgency.includes("URGENT") || urgency.includes("CRITICAL")
                      ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
                      : urgency.includes("HIGH") || urgency.includes("MODERATE")
                      ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
                      : "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                  }`}
                >
                  SEVERITY: {urgency}
                </span>

                <button
                  type="button"
                  onClick={copyReportText}
                  className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium transition"
                  title="Copy Report to Clipboard"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? "Copied" : "Copy Report"}
                </button>
              </div>
            </div>

            <div className="space-y-3 text-xs text-slate-300">
              {fullMarkdown ? (
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800/80 leading-relaxed text-slate-200 font-mono text-xs whitespace-pre-wrap">
                  {fullMarkdown}
                </div>
              ) : (
                <>
                  {summary && (
                    <div>
                      <h5 className="font-semibold text-slate-400 uppercase tracking-wider text-[10px] mb-1">
                        Clinical Indication & Summary
                      </h5>
                      <p className="bg-slate-950 p-3 rounded-lg border border-slate-800/80 leading-relaxed text-slate-200">
                        {summary}
                      </p>
                    </div>
                  )}

                  {impression && (
                    <div>
                      <h5 className="font-semibold text-slate-400 uppercase tracking-wider text-[10px] mb-1">
                        Impression
                      </h5>
                      <p className="bg-slate-950 p-3 rounded-lg border border-slate-800/80 font-medium text-slate-200 leading-relaxed">
                        {impression}
                      </p>
                    </div>
                  )}

                  {recommendations.length > 0 && (
                    <div>
                      <h5 className="font-semibold text-slate-400 uppercase tracking-wider text-[10px] mb-1">
                        Recommendations
                      </h5>
                      <ul className="list-disc list-inside space-y-1 bg-slate-950 p-3 rounded-lg border border-slate-800/80 text-slate-300">
                        {recommendations.map((rec: string, idx: number) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}

              <div className="flex items-center gap-1.5 pt-1 text-[11px] text-slate-500">
                <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                <span>{disclaimer}</span>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
