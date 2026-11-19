"use client";

import React, { useRef, useState } from "react";
import { Download, FileText, CheckCircle2, AlertTriangle } from "lucide-react";
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
        backgroundColor: "#020617",
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
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">
            <AlertTriangle className="w-3.5 h-3.5" />
            Urgent Attention
          </span>
        );
      case "moderate":
      case "high":
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
            <AlertTriangle className="w-3.5 h-3.5" />
            Moderate Findings
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Low Risk / Stable
          </span>
        );
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Action Bar */}
      <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 px-4 py-3 rounded-xl">
        <div className="flex items-center gap-2.5">
          <FileText className="w-4 h-4 text-blue-400" />
          <h3 className="font-semibold text-slate-100 text-sm">
            Radiology Diagnostic Report
          </h3>
        </div>

        <div className="flex items-center gap-3">
          {getSeverityBadge(report.severity_level || "moderate")}
          <button
            type="button"
            onClick={handleExportPDF}
            disabled={exporting}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{exporting ? "Generating PDF..." : "Export PDF"}</span>
          </button>
        </div>
      </div>

      {/* Printable Report Document Card */}
      <div
        ref={reportRef}
        className="bg-slate-950 border border-slate-800 rounded-xl p-6 shadow-xl text-slate-200"
      >
        {/* Document Header */}
        <div className="border-b border-slate-800 pb-4 mb-5 flex items-start justify-between">
          <div>
            <h1 className="text-base font-bold text-slate-100 tracking-tight">
              Chest Radiograph Clinical Report
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Automated computer-aided radiograph analysis
            </p>
          </div>
          <div className="text-right text-xs text-slate-400 space-y-0.5">
            <div>
              <span className="text-slate-500">Scan ID: </span>
              <span className="font-medium text-slate-300">{patientId || "CXRAY-SCAN"}</span>
            </div>
            <div>
              <span className="text-slate-500">Date: </span>
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
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 mb-5">
            <span className="text-[11px] font-semibold text-blue-300 block mb-1">
              Clinical Impression Summary
            </span>
            <p className="text-xs text-slate-200 leading-relaxed">
              {report.summary}
            </p>
          </div>
        )}

        {/* Formatted Markdown Body */}
        <div className="text-xs text-slate-300 leading-relaxed space-y-3 whitespace-pre-wrap font-sans">
          {report.report}
        </div>

        {/* Recommendations */}
        {report.recommendations && report.recommendations.length > 0 && (
          <div className="mt-5 pt-4 border-t border-slate-800">
            <span className="text-xs font-semibold text-slate-300 block mb-2">
              Clinical Recommendations
            </span>
            <ul className="space-y-1.5">
              {report.recommendations.map((rec, i) => (
                <li key={i} className="text-xs text-slate-300 flex items-start gap-2">
                  <span className="text-blue-400 font-bold">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Footer Disclaimer */}
        <div className="border-t border-slate-800 pt-4 mt-6 flex items-center justify-between text-[11px] text-slate-500">
          <span>
            Research prototype for computer-aided chest X-ray interpretation.
          </span>
          <span className="font-medium text-slate-400">
            Requires Certified Radiologist Review
          </span>
        </div>
      </div>
    </div>
  );
};
