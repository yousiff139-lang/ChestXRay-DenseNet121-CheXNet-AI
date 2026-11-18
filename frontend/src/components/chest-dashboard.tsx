"use client";

import React, { useState, useEffect } from "react";
import { useChestClassification } from "../hooks/use-chest-classification";
import { XRayDropzone } from "./xray-dropzone";
import { BoundingBoxCanvas } from "./bounding-box-canvas";
import { DiseaseProbabilityChart } from "./disease-probability-chart";
import { CurativeTreatmentCard } from "./results/curative-treatment-card";
import { DiagnosticReportViewer } from "./results/diagnostic-report-viewer";
import { ModelComparison } from "./model-comparison";
import {
  Layers,
  BarChart3,
  BookOpen,
  RotateCcw,
  Cpu,
  Award,
  AlertCircle,
  Stethoscope,
  Scan,
  ShieldCheck,
  Activity,
  FileText,
  Square,
  Sparkles,
  CheckCircle2,
  Clock,
} from "lucide-react";

export const ChestDashboard: React.FC = () => {
  const [activeMainTab, setActiveMainTab] = useState<"analysis" | "benchmark" | "about">("analysis");
  const [activeResultTab, setActiveResultTab] = useState<
    "overview" | "detections" | "curative" | "report" | "comparison"
  >("overview");
  const [selectedModelMode, setSelectedModelMode] = useState<"unified" | "densenet" | "resnet" | "both">("unified");

  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Overlay Controls for Radiography Canvas
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showGradcam, setShowGradcam] = useState<boolean>(true);
  const [gradcamOpacity, setGradcamOpacity] = useState<number>(50);

  const {
    loading,
    comparing,
    diagnosing,
    error,
    classificationResult,
    comparisonResult,
    unifiedResult,
    benchmarkData,
    detectAndDiagnose,
    classifyImage,
    compareModels,
    fetchBenchmark,
    clearResults,
  } = useChestClassification();

  // Load benchmark metrics on mount
  useEffect(() => {
    fetchBenchmark();
  }, [fetchBenchmark]);

  // Handle file drop/upload
  const handleFileSelect = async (file: File) => {
    setCurrentFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    if (selectedModelMode === "unified") {
      await detectAndDiagnose(file, true, true);
    } else if (selectedModelMode === "both") {
      await compareModels(file);
    } else {
      await classifyImage(file, selectedModelMode);
    }
  };

  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setCurrentFile(null);
    setPreviewUrl(null);
    clearResults();
    setActiveResultTab("overview");
  };

  const isBusy = loading || comparing || diagnosing;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Clinical Navigation Bar */}
      <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur border-b border-slate-800 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-600/20 to-blue-600/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shadow-inner">
              <Scan className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold tracking-tight text-white">
                  ChestXRay AI Diagnostic Workstation
                </h1>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  DenseNet + YOLO + Gemini VLM
                </span>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Seminar 16
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                DenseNet-121 Multi-Label Classification, YOLO Lesion Localization & Gemini Curative Protocol Reporting
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 p-1 rounded-lg text-xs">
            <button
              type="button"
              onClick={() => setActiveMainTab("analysis")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                activeMainTab === "analysis"
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Stethoscope className="w-3.5 h-3.5" />
              Diagnostic Workstation
            </button>
            <button
              type="button"
              onClick={() => setActiveMainTab("benchmark")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                activeMainTab === "benchmark"
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Model Benchmark
            </button>
            <button
              type="button"
              onClick={() => setActiveMainTab("about")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
                activeMainTab === "about"
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              Seminar Spec
            </button>
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
                New Examination
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-6 space-y-6">
        {/* Error Alert */}
        {error && (
          <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <div>
              <p className="font-semibold">Diagnostic Service Error</p>
              <p className="text-rose-300/80">{error}</p>
            </div>
          </div>
        )}

        {/* Tab 1: Diagnostic Workstation */}
        {activeMainTab === "analysis" && (
          <div className="space-y-6">
            {/* Model Selector Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center gap-3 text-xs">
                <span className="font-medium text-slate-400 uppercase tracking-wider text-[10px]">
                  Pipeline Mode:
                </span>
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedModelMode("unified");
                      if (currentFile) detectAndDiagnose(currentFile, true, true);
                    }}
                    className={`px-3 py-1 rounded text-xs transition flex items-center gap-1.5 ${
                      selectedModelMode === "unified"
                        ? "bg-cyan-600 text-white font-medium shadow-sm"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    <Sparkles className="w-3.5 h-3.5 text-cyan-200" /> Unified (DenseNet + YOLO + VLM)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedModelMode("densenet");
                      if (currentFile) classifyImage(currentFile, "densenet");
                    }}
                    className={`px-3 py-1 rounded text-xs transition flex items-center gap-1.5 ${
                      selectedModelMode === "densenet"
                        ? "bg-blue-600 text-white font-medium"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    <Award className="w-3.5 h-3.5" /> DenseNet-121 (Primary)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedModelMode("resnet");
                      if (currentFile) classifyImage(currentFile, "resnet");
                    }}
                    className={`px-3 py-1 rounded text-xs transition flex items-center gap-1.5 ${
                      selectedModelMode === "resnet"
                        ? "bg-slate-700 text-white font-medium"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    <Cpu className="w-3.5 h-3.5" /> ResNet-50 (Baseline)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedModelMode("both");
                      if (currentFile) compareModels(currentFile);
                    }}
                    className={`px-3 py-1 rounded text-xs transition flex items-center gap-1.5 ${
                      selectedModelMode === "both"
                        ? "bg-slate-700 text-white font-medium"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" /> Dual Comparison
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>NIH ChestX-ray14 Calibrated Weights</span>
              </div>
            </div>

            {/* Dropzone if no file selected yet */}
            {!previewUrl && (
              <div className="max-w-2xl mx-auto py-8">
                <XRayDropzone onFileSelect={handleFileSelect} disabled={isBusy} />
              </div>
            )}

            {/* Multi-Stage Loading state indicator */}
            {isBusy && (
              <div className="p-10 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col items-center justify-center text-center shadow-xl">
                <div className="w-10 h-10 rounded-full border-2 border-cyan-500/30 border-t-cyan-500 animate-spin mb-4" />
                <h4 className="text-sm font-semibold text-slate-100">
                  Executing Comprehensive Thoracic AI Diagnostic Pipeline...
                </h4>
                <div className="flex flex-wrap items-center justify-center gap-2 mt-4 max-w-xl text-[11px] text-slate-400">
                  <span className="px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 flex items-center gap-1 text-blue-300">
                    <Award className="w-3 h-3" /> DenseNet-121 Classification
                  </span>
                  <span className="px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 flex items-center gap-1 text-purple-300">
                    <Square className="w-3 h-3" /> YOLO Pathology Bounding Boxes
                  </span>
                  <span className="px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 flex items-center gap-1 text-cyan-300">
                    <Sparkles className="w-3 h-3" /> Gemini VLM Curative Dossier
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-3 font-mono">
                  Calibrating 14 multi-label outputs, generating Grad-CAM heatmaps, and computing curative regimens.
                </p>
              </div>
            )}

            {/* Diagnostic Results Presentation */}
            {previewUrl && !isBusy && (
              <div className="space-y-6">
                {/* Secondary Results View Tabs */}
                <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/60 p-2 rounded-xl border border-slate-800">
                  <div className="flex items-center gap-1 text-xs">
                    <button
                      type="button"
                      onClick={() => setActiveResultTab("overview")}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
                        activeResultTab === "overview"
                          ? "bg-cyan-600 text-white"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <Scan className="w-3.5 h-3.5" />
                      Visual Radiology & Localization
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveResultTab("curative")}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
                        activeResultTab === "curative"
                          ? "bg-cyan-600 text-white"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <Activity className="w-3.5 h-3.5" />
                      Curative Protocols
                      {unifiedResult?.curative_protocols && (
                        <span className="ml-1 px-1.5 py-0.2 rounded-full text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
                          {unifiedResult.curative_protocols.length}
                        </span>
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveResultTab("report")}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
                        activeResultTab === "report"
                          ? "bg-cyan-600 text-white"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <FileText className="w-3.5 h-3.5" />
                      Radiology Report Dossier
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveResultTab("comparison")}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition ${
                        activeResultTab === "comparison"
                          ? "bg-cyan-600 text-white"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <Layers className="w-3.5 h-3.5" />
                      Dual CNN Comparison
                    </button>
                  </div>

                  {/* Latency & Findings Count Badge */}
                  <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                    {unifiedResult && (
                      <>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5 text-cyan-400" />
                          <span>End-to-end: {unifiedResult.end_to_end_ms.toFixed(0)}ms</span>
                        </span>
                        <span className="flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                          <span>{unifiedResult.detections.length} Lesions Squared</span>
                        </span>
                      </>
                    )}
                  </div>
                </div>

                {/* Sub-view 1: Overview (Visual Canvas + Detections + NIH-14 Probabilities) */}
                {activeResultTab === "overview" && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                      {/* Left: Interactive Bounding Box & Grad-CAM Canvas (7 cols) */}
                      <div className="lg:col-span-7">
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

                      {/* Right: Disease Probabilities (5 cols) */}
                      <div className="lg:col-span-5 space-y-4">
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

                        {/* Localized Pathology Summary Cards */}
                        {unifiedResult && unifiedResult.detections.length > 0 && (
                          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
                            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                              <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                                <Square className="w-3.5 h-3.5 text-cyan-400" />
                                Localized Pathology Bounding Boxes
                              </span>
                              <span className="text-[10px] font-mono text-cyan-400">
                                {unifiedResult.detections.length} region(s)
                              </span>
                            </div>

                            <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                              {unifiedResult.detections.map((det) => (
                                <div
                                  key={det.id}
                                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 text-xs"
                                >
                                  <div className="flex items-center gap-2">
                                    <span
                                      className="w-2.5 h-2.5 rounded-full"
                                      style={{ backgroundColor: det.color }}
                                    />
                                    <div>
                                      <span className="font-semibold text-slate-100">
                                        {det.class_name}
                                      </span>
                                      <p className="text-[10px] text-slate-400">
                                        Box: [{Math.round(det.bbox.x)}, {Math.round(det.bbox.y)}, {Math.round(det.bbox.width)}×{Math.round(det.bbox.height)}]
                                      </p>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <span className="font-mono text-xs text-cyan-300 font-semibold">
                                      {(det.confidence * 100).toFixed(1)}%
                                    </span>
                                    <span
                                      className={`block text-[9px] uppercase font-bold tracking-wider ${
                                        det.severity === "severe"
                                          ? "text-rose-400"
                                          : det.severity === "moderate"
                                          ? "text-amber-400"
                                          : "text-emerald-400"
                                      }`}
                                    >
                                      {det.severity}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Quick Curative Regimens Preview */}
                    {unifiedResult?.curative_protocols && unifiedResult.curative_protocols.length > 0 && (
                      <div className="mt-4">
                        <CurativeTreatmentCard protocols={unifiedResult.curative_protocols} />
                      </div>
                    )}
                  </div>
                )}

                {/* Sub-view 2: Curative & Treatment Protocols */}
                {activeResultTab === "curative" && (
                  <div className="space-y-4">
                    <CurativeTreatmentCard
                      protocols={
                        unifiedResult?.curative_protocols || []
                      }
                    />
                  </div>
                )}

                {/* Sub-view 3: Full Diagnostic Dossier Report */}
                {activeResultTab === "report" && (
                  <div className="space-y-4">
                    {unifiedResult?.diagnostic_report ? (
                      <DiagnosticReportViewer
                        report={unifiedResult.diagnostic_report}
                        patientId="PT-CHESTXRAY-14"
                      />
                    ) : (
                      <div className="p-8 text-center bg-slate-900/60 border border-slate-800 rounded-xl text-slate-400 text-xs">
                        No report available yet. Run a unified diagnosis to generate the full dossier.
                      </div>
                    )}
                  </div>
                )}

                {/* Sub-view 4: Dual CNN Comparison (DenseNet-121 vs ResNet-50) */}
                {activeResultTab === "comparison" && (
                  <div className="space-y-4">
                    {comparisonResult ? (
                      <ModelComparison
                        comparisonResult={comparisonResult}
                        benchmarkData={benchmarkData}
                      />
                    ) : (
                      <div className="p-8 text-center bg-slate-900/60 border border-slate-800 rounded-xl text-slate-400 text-xs">
                        Comparison data not loaded. Switch to &quot;Dual Comparison&quot; mode to run side-by-side inference.
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Model Benchmark */}
        {activeMainTab === "benchmark" && (
          <ModelComparison
            comparisonResult={comparisonResult}
            benchmarkData={benchmarkData}
          />
        )}

        {/* Tab 3: About Seminar 16 Spec */}
        {activeMainTab === "about" && (
          <div className="bg-slate-900/60 rounded-xl border border-slate-800 p-6 max-w-4xl mx-auto space-y-5 text-xs text-slate-300">
            <div className="border-b border-slate-800 pb-3">
              <span className="text-cyan-400 font-mono font-semibold uppercase tracking-wider text-[10px]">
                Syllabus & Clinical Specification
              </span>
              <h2 className="text-base font-semibold text-white mt-1">
                Seminar 16: Chest X-Ray Disease Classification & Multi-Stage Diagnostic AI
              </h2>
            </div>

            <div className="space-y-3 leading-relaxed">
              <p>
                <strong>Seminar Objective:</strong> Develop a multi-stage chest X-ray classification and lesion localization system combining DenseNet-121 multi-label inference, YOLO thoracic pathology localization (&quot;squaring the lesions&quot;), and Gemini Vision VLM clinical curative reporting.
              </p>
              <p>
                <strong>Evaluation Criteria (Mandatory 4 Metrics):</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2 text-slate-200">
                <li><strong>(1) Mean ROC-AUC:</strong> Area under the ROC curve aggregated across all 14 clinical classes (DenseNet-121: 0.841 vs ResNet-50: 0.812).</li>
                <li><strong>(2) Mean Average Precision (mAP):</strong> Precision-recall curve area across imbalanced medical categories.</li>
                <li><strong>(3) Per-class Sensitivity:</strong> True Positive Rate ($TP / (TP + FN)$) at calibrated decision boundaries.</li>
                <li><strong>(4) Inference Time:</strong> Wall-clock forward pass latency ($ms$) for real-time deployment feasibility.</li>
              </ul>
              <p>
                <strong>Visual Interpretability & Localization:</strong> Disease-relevant thoracic regions are visualized using both Grad-CAM class activation mapping on DenseNet&apos;s final convolutional layer (<code className="text-slate-300">features[-1]</code>) and YOLO bounding box squares with calibrated confidence indicators.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 space-y-1">
              <h4 className="font-semibold text-slate-200">Why DenseNet-121 Outperforms Standard ResNets</h4>
              <p className="text-slate-400">
                In chest radiography, subtle opacities (e.g. pneumothorax, pulmonary infiltrates) involve fine spatial gradients. DenseNet-121 connects each layer to every subsequent layer, preserving subtle low-level feature maps across depths while requiring only 7.98M parameters compared to ResNet-50&apos;s 25.56M parameters (3.2× more parameter-efficient).
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};