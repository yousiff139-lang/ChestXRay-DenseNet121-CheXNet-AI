"use client";

import React, { useState } from "react";
import { useChestClassification } from "../hooks/use-chest-classification";
import { XRayDropzone } from "./xray-dropzone";
import { BoundingBoxCanvas } from "./bounding-box-canvas";
import { DiseaseProbabilityChart } from "./disease-probability-chart";
import { CurativeTreatmentCard } from "./results/curative-treatment-card";
import { DiagnosticReportViewer } from "./results/diagnostic-report-viewer";
import { ModelComparison } from "./model-comparison";
import {
  RotateCcw,
  AlertCircle,
  FileText,
  Activity,
  Layers,
  Image as ImageIcon,
  Loader2,
  CheckCircle2,
  Stethoscope,
} from "lucide-react";

export const ChestDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"findings" | "protocols" | "report" | "comparison">("findings");

  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Overlay Controls for Radiograph Canvas
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showGradcam, setShowGradcam] = useState<boolean>(true);
  const [gradcamOpacity, setGradcamOpacity] = useState<number>(50);

  const {
    loading,
    comparing,
    diagnosing,
    reporting,
    error,
    classificationResult,
    comparisonResult,
    unifiedResult,
    benchmarkData,
    detectAndDiagnose,
    generateReportForImage,
    compareModels,
    fetchBenchmark,
    clearResults,
  } = useChestClassification();

  // Fast upload: Only classifies and localizes bounding boxes immediately
  const handleFileSelect = async (file: File) => {
    setCurrentFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setActiveTab("findings");

    // Fast diagnosis without waiting for long text report
    await detectAndDiagnose(file, true, false);
  };

  // Dedicated button to generate clinical report on demand
  const handleGenerateReport = async () => {
    if (!currentFile) return;
    await generateReportForImage(currentFile);
    setActiveTab("report");
  };

  const handleTabChange = (tab: "findings" | "protocols" | "report" | "comparison") => {
    setActiveTab(tab);
    if (tab === "comparison" && !benchmarkData) {
      fetchBenchmark();
    }
  };

  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setCurrentFile(null);
    setPreviewUrl(null);
    clearResults();
    setActiveTab("findings");
  };

  const isAnalyzing = diagnosing || loading;
  const isGeneratingReport = reporting;
  const hasResults = !!(unifiedResult || classificationResult);
  const hasReport = !!unifiedResult?.diagnostic_report;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Top Clinical Navigation Bar */}
      <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold tracking-tight text-white">
                  Chest Radiography Workstation
                </h1>
                <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  DenseNet-121 &amp; YOLO Localization
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                14-pathology multi-label classification and lesion bounding box localization
              </p>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {previewUrl && (
              <button
                type="button"
                onClick={handleReset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-medium transition"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                New Scan
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 pt-6 pb-20 space-y-6">
        {/* Error Alert */}
        {error && (
          <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <div>
              <p className="font-semibold">Analysis Notice</p>
              <p className="text-rose-300/80">{error}</p>
            </div>
          </div>
        )}

        {/* Dropzone if no file selected yet */}
        {!previewUrl && !isAnalyzing && (
          <div className="max-w-2xl mx-auto py-8">
            <XRayDropzone onFileSelect={handleFileSelect} disabled={isAnalyzing} />
          </div>
        )}

        {/* Fast Analysis Loading State */}
        {isAnalyzing && (
          <div className="py-24 rounded-2xl bg-slate-900/50 border border-slate-800 flex flex-col items-center justify-center text-center">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin mb-3" />
            <h4 className="text-sm font-semibold text-slate-200">
              Analyzing Chest Radiograph...
            </h4>
            <p className="text-xs text-slate-400 mt-1 max-w-sm">
              Computing DenseNet-121 multi-label classifications and localizing thoracic lesions.
            </p>
          </div>
        )}

        {/* Diagnostic Results Presentation */}
        {previewUrl && hasResults && !isAnalyzing && (
          <div className="space-y-5">
            {/* Navigation Tabs */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-2">
              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  type="button"
                  onClick={() => handleTabChange("findings")}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                    activeTab === "findings"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <ImageIcon className="w-3.5 h-3.5" />
                  Image &amp; Findings
                </button>

                <button
                  type="button"
                  onClick={() => handleTabChange("protocols")}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                    activeTab === "protocols"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" />
                  Treatment Guidelines
                  {unifiedResult?.curative_protocols && unifiedResult.curative_protocols.length > 0 && (
                    <span className="ml-1 px-1.5 py-0.2 rounded-full text-[10px] bg-blue-950 text-blue-300 border border-blue-800">
                      {unifiedResult.curative_protocols.length}
                    </span>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => handleTabChange("report")}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                    activeTab === "report"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  Clinical Report
                  {hasReport && (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block ml-0.5" />
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => handleTabChange("comparison")}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                    activeTab === "comparison"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  Model Comparison
                </button>
              </div>

              {/* Quick Action: Generate Report Button */}
              <div className="flex items-center gap-2">
                {!hasReport ? (
                  <button
                    type="button"
                    onClick={handleGenerateReport}
                    disabled={isGeneratingReport}
                    className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition disabled:opacity-50 shadow-sm"
                  >
                    {isGeneratingReport ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <FileText className="w-3.5 h-3.5" />
                    )}
                    <span>{isGeneratingReport ? "Writing Report..." : "Generate Clinical Report"}</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setActiveTab("report")}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-950/60 text-emerald-300 border border-emerald-800/60 hover:bg-emerald-900/60 transition"
                  >
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Report Ready (View)</span>
                  </button>
                )}
              </div>
            </div>

            {/* TAB 1: Image & Findings */}
            {activeTab === "findings" && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                {/* Left: Interactive Canvas with Bounding Boxes & GradCAM (7 cols) */}
                <div className="lg:col-span-7 space-y-3">
                  <BoundingBoxCanvas
                    imageSrc={previewUrl}
                    gradcamSrc={
                      unifiedResult?.gradcam_overlay_base64 ||
                      classificationResult?.gradcam_overlay_base64
                    }
                    detections={unifiedResult?.detections || []}
                    showBoxes={showBoundingBoxes}
                    showGradcam={showGradcam}
                    gradcamOpacity={gradcamOpacity}
                    onToggleBoxes={setShowBoundingBoxes}
                    onToggleGradcam={setShowGradcam}
                    onChangeOpacity={setGradcamOpacity}
                  />
                </div>

                {/* Right: Detected Regions + Multi-Label Probabilities (5 cols) */}
                <div className="lg:col-span-5 flex flex-col gap-4">
                  {/* Localized Lesion Summary (if boxes exist) */}
                  {unifiedResult && unifiedResult.detections.length > 0 ? (
                    <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5 space-y-2.5">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <span className="text-xs font-semibold text-slate-200">
                          Localized Lesions ({unifiedResult.detections.length})
                        </span>
                        <span className="text-[11px] text-slate-400">
                          Squared regions on image
                        </span>
                      </div>

                      <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
                        {unifiedResult.detections.map((det) => (
                          <div
                            key={det.id}
                            className="flex items-center justify-between p-2 rounded-lg bg-slate-950/80 border border-slate-800 text-xs"
                          >
                            <div className="flex items-center gap-2">
                              <span
                                className="w-2.5 h-2.5 rounded-full"
                                style={{ backgroundColor: det.color }}
                              />
                              <span className="font-semibold text-slate-100">
                                {det.class_name}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-semibold text-slate-200">
                                {(det.confidence * 100).toFixed(0)}%
                              </span>
                              <span
                                className={`text-[10px] font-medium px-1.5 py-0.5 rounded capitalize ${
                                  det.severity === "severe"
                                    ? "bg-rose-950/60 text-rose-300 border border-rose-800/50"
                                    : det.severity === "moderate"
                                    ? "bg-amber-950/60 text-amber-300 border border-amber-800/50"
                                    : "bg-emerald-950/60 text-emerald-300 border border-emerald-800/50"
                                }`}
                              >
                                {det.severity}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-3 text-xs text-slate-400 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>No focal lesions requiring bounding box localization.</span>
                    </div>
                  )}

                  {/* Classification Probability Chart */}
                  <DiseaseProbabilityChart
                    predictions={
                      unifiedResult?.predictions ||
                      classificationResult?.predictions ||
                      []
                    }
                    confidenceThreshold={
                      classificationResult?.confidence_threshold || 0.5
                    }
                  />

                  {/* Primary CTA Card */}
                  <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold text-slate-200">
                        {hasReport ? "Clinical Report Available" : "Need a Full Medical Report?"}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        {hasReport
                          ? "Radiologist findings, diagnosis, and recommendations ready."
                          : "Generate a comprehensive structured clinical report on demand."}
                      </p>
                    </div>

                    {!hasReport ? (
                      <button
                        type="button"
                        onClick={handleGenerateReport}
                        disabled={isGeneratingReport}
                        className="shrink-0 flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition disabled:opacity-50"
                      >
                        {isGeneratingReport ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <FileText className="w-3.5 h-3.5" />
                        )}
                        <span>{isGeneratingReport ? "Generating..." : "Generate"}</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setActiveTab("report")}
                        className="shrink-0 px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
                      >
                        View Report
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: Curative Treatment Guidelines */}
            {activeTab === "protocols" && (
              <div className="space-y-4">
                <CurativeTreatmentCard
                  protocols={unifiedResult?.curative_protocols || []}
                />
              </div>
            )}

            {/* TAB 3: Clinical Report */}
            {activeTab === "report" && (
              <div className="space-y-4">
                {unifiedResult?.diagnostic_report ? (
                  <DiagnosticReportViewer
                    report={unifiedResult.diagnostic_report}
                    patientId="SCAN-CHEST"
                  />
                ) : (
                  <div className="py-16 text-center bg-slate-900/50 border border-slate-800 rounded-2xl max-w-xl mx-auto p-8 space-y-4">
                    <div className="w-12 h-12 rounded-xl bg-blue-600/10 border border-blue-500/20 text-blue-400 flex items-center justify-center mx-auto">
                      <FileText className="w-6 h-6" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-slate-200">
                        No Clinical Report Generated Yet
                      </h3>
                      <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                        Click below to generate a structured medical report with clinical impression, findings, and follow-up guidance.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={handleGenerateReport}
                      disabled={isGeneratingReport}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition disabled:opacity-50"
                    >
                      {isGeneratingReport ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <FileText className="w-4 h-4" />
                      )}
                      <span>{isGeneratingReport ? "Writing Report..." : "Generate Clinical Report"}</span>
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* TAB 4: Model Comparison */}
            {activeTab === "comparison" && (
              <div className="space-y-4">
                {comparisonResult ? (
                  <ModelComparison
                    comparisonResult={comparisonResult}
                    benchmarkData={benchmarkData}
                  />
                ) : (
                  <div className="py-16 text-center bg-slate-900/50 border border-slate-800 rounded-2xl max-w-md mx-auto p-8 space-y-3">
                    <Layers className="w-8 h-8 text-slate-500 mx-auto" />
                    <p className="text-xs text-slate-400">
                      Run dual-model inference to compare DenseNet-121 against ResNet-50.
                    </p>
                    <button
                      type="button"
                      onClick={() => currentFile && compareModels(currentFile)}
                      disabled={comparing}
                      className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
                    >
                      {comparing && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                      Compare DenseNet-121 vs ResNet-50
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto w-full border-t border-slate-800/80 py-5 px-6 text-center text-xs text-slate-500 bg-slate-950 relative z-10">
        Research prototype for thoracic pathology detection and lesion localization. Requires certified radiologist verification.
      </footer>
    </div>
  );
};
