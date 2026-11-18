"use client";

import React, { useRef, useState } from "react";
import { Download, FileText, CheckCircle2, AlertTriangle, ShieldAlert } from "lucide-react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

interface DiagnosticReportViewerProps {
  report: {
    report: string;
    summary: string;
    recommendations: string[];
    severity_level: string;
    generated_at?: string;
  };
  patientId?: string;
}

export const DiagnosticReportViewer: React.FC<DiagnosticReportViewerProps> = ({
  report,
  patientId,
}) => {
  const reportRef = useRef<HTMLDivElement>(null);
  const [exporting, setExporting] = useState<boolean>(false);

  const handleExportPDF = async () => {
    if (!reportRef.current) return;
    setExporting(true);

    try {
      const element = reportRef.current;
      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#090d16",
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "pt",
        format: "a4",
      });

      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(`ChestXRay_Radiology_Report_${patientId || "Scan"}.pdf`);
    } catch (err) {
      console.error("PDF export failed:", err);
    } finally {
      setExporting(false);
    }
  };

  const getSeverityBadge = (level: string) => {
    switch (level.toLowerCase()) {
      case "severe":
      case "urgent":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
            <ShieldAlert className="w-3.5 h-3.5" />
            Severe / Urgent
          </span>
        );
      case "moderate":
      case "high":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
            <AlertTriangle className="w-3.5 h-3.5" />
            Moderate Severity
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Normal / Low Risk
          </span>
        );
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Action Bar */}
      <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 px-5 py-3 rounded-2xl">
        <div className="flex items-center gap-3">
          <FileText className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="font-bold text-slate-100 text-sm">
              Comprehensive Radiologist Clinical Dossier
            </h3>
            <p className="text-[11px] text-slate-400">
              Generated via Gemini Multimodal VLM with Curative Protocol Synthesis
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {getSeverityBadge(report.severity_level || "moderate")}
          <button
            type="button"
            onClick={handleExportPDF}
            disabled={exporting}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition-all shadow-md active:scale-95 disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{exporting ? "Generating PDF..." : "Export PDF Dossier"}</span>
          </button>
        </div>
      </div>

      {/* Printable Report Document Card */}
      <div
        ref={reportRef}
        className="bg-slate-950 border border-slate-800/90 rounded-2xl p-7 shadow-2xl text-slate-200"
      >
        {/* Document Header */}
        <div className="border-b border-slate-800 pb-5 mb-6 flex items-start justify-between">
          <div>
            <h1 className="text-lg font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <span className="text-cyan-400">🫁</span>
              CHEST X-RAY RADIOLOGY CLINICAL REPORT
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Dual-Brain Artificial Intelligence Radiography Interpretation System
            </p>
          </div>
          <div className="text-right text-xs text-slate-400 space-y-1">
            <div>
              <span className="text-slate-500">Case ID: </span>
              <span className="font-mono text-slate-300">{patientId || "ANON-CXRAY"}</span>
            </div>
            <div>
              <span className="text-slate-500">Report Date: </span>
              <span className="text-slate-300">
                {report.generated_at
                  ? new Date(report.generated_at).toLocaleString()
                  : new Date().toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>

        {/* Clinical Summary Highlight Box */}
        {report.summary && (
          <div className="bg-cyan-950/20 border border-cyan-800/30 rounded-xl p-4 mb-6">
            <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider block mb-1">
              Executive Clinical Impression
            </span>
            <p className="text-xs text-cyan-100/90 leading-relaxed font-medium">
              {report.summary}
            </p>
          </div>
        )}

        {/* Formatted Markdown Body */}
        <div className="text-xs text-slate-300 leading-relaxed space-y-4 whitespace-pre-wrap font-sans">
          {report.report}
        </div>

        {/* Footer Disclaimer */}
        <div className="border-t border-slate-800/80 pt-5 mt-7 flex items-center justify-between text-[11px] text-slate-500">
          <span>
            Research prototype developed for thoracic disease detection and localization.
          </span>
          <span className="font-semibold text-slate-400">
            Requires Certified Radiologist Attestation
          </span>
        </div>
      </div>
    </div>
  );
};
