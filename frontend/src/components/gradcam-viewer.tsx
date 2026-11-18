/* eslint-disable @next/next/no-img-element */
"use client";

import React, { useState } from "react";
import { Eye, EyeOff, Layers, ZoomIn, ZoomOut, RotateCcw, Contrast } from "lucide-react";

interface GradCamViewerProps {
  originalImageUrl: string;
  gradcamBase64?: string | null;
  gradcamClassName?: string | null;
  modelUsed?: string;
  inferenceTimeMs?: number;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({
  originalImageUrl,
  gradcamBase64,
  gradcamClassName,
  modelUsed = "DenseNet-121",
  inferenceTimeMs,
}) => {
  const [opacity, setOpacity] = useState<number>(65);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [invert, setInvert] = useState<boolean>(false);
  const [zoom, setZoom] = useState<number>(1);

  const resetView = () => {
    setZoom(1);
    setOpacity(65);
    setShowOverlay(true);
    setInvert(false);
  };

  const isNoEvidence = !gradcamBase64 || gradcamClassName === "No evidence";
  const targetLayer = modelUsed === "ResNet-50" ? "layer4[-1]" : "features.norm5";

  return (
    <div className="flex flex-col h-full bg-slate-900/60 rounded-xl border border-slate-800 p-4">
      {/* Viewport Header Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-slate-100">
              Radiological Viewport & Grad-CAM
            </h4>
            <span className="px-2 py-0.5 text-[10px] font-mono font-medium rounded bg-slate-800 text-slate-300 border border-slate-700">
              {modelUsed}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Activation mapping on <code className="text-slate-300 font-mono">{targetLayer}</code>
            {gradcamClassName && gradcamClassName !== "No evidence" && (
              <span className="ml-1.5 text-rose-400 font-medium">
                • Explaining: {gradcamClassName.replace("_", " ")}
              </span>
            )}
          </p>
        </div>

        {/* Diagnostic Meta */}
        {inferenceTimeMs !== undefined && (
          <div className="flex items-center gap-3 text-xs">
            <div className="px-2.5 py-1 rounded bg-slate-950 border border-slate-800 flex items-center gap-1.5 font-mono text-[11px]">
              <span className="text-slate-500">LATENCY:</span>
              <span className="text-blue-400 font-semibold">{inferenceTimeMs.toFixed(1)} ms</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Image Stage */}
      <div className="relative flex-1 min-h-[380px] bg-black rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center select-none">
        {/* Subtle PACS alignment grid */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:16px_16px]" />

        <div
          className="relative transition-transform duration-150 ease-out"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Base X-Ray Image */}
          <img
            src={originalImageUrl}
            alt="Chest Radiograph"
            className="max-h-[380px] w-auto object-contain transition-all"
            style={{ filter: invert ? "invert(1)" : "none" }}
          />

          {/* Grad-CAM Heatmap Overlay */}
          {gradcamBase64 && showOverlay && (
            <img
              src={`data:image/jpeg;base64,${gradcamBase64}`}
              alt="Grad-CAM Activation Heatmap"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen transition-opacity duration-150"
              style={{ opacity: opacity / 100 }}
            />
          )}
        </div>

        {/* Notice when no positive finding */}
        {isNoEvidence && (
          <div className="absolute top-3 left-3 right-3 bg-slate-950/85 backdrop-blur border border-slate-800 text-slate-300 text-[11px] px-3 py-1.5 rounded flex items-center justify-between">
            <span>No evidence above validation threshold — all 14 thoracic findings negative.</span>
            <span className="text-slate-500 text-[10px] uppercase font-mono">Heatmap suppressed</span>
          </div>
        )}

        {/* Top-Right HUD Badge */}
        <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 bg-slate-950/80 backdrop-blur border border-slate-800 rounded px-2 py-1 text-[10px] font-mono text-slate-400">
          <span>TARGET: {targetLayer}</span>
        </div>

        {/* Viewport Action Controls */}
        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-2 pointer-events-auto">
          {/* Tool toggles */}
          <div className="flex items-center gap-1 bg-slate-950/90 backdrop-blur border border-slate-800 rounded-lg p-1 text-xs">
            <button
              type="button"
              onClick={() => setShowOverlay((v) => !v)}
              className={`px-2.5 py-1 rounded flex items-center gap-1 font-medium text-[11px] transition ${
                showOverlay
                  ? "bg-blue-600 text-white"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              }`}
              title="Toggle Heatmap"
            >
              {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              Heatmap
            </button>

            <button
              type="button"
              onClick={() => setInvert((v) => !v)}
              className={`px-2.5 py-1 rounded flex items-center gap-1 font-medium text-[11px] transition ${
                invert
                  ? "bg-slate-700 text-white"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              }`}
              title="Invert Grayscale (Radiological Negative)"
            >
              <Contrast className="w-3.5 h-3.5" />
              Invert
            </button>
          </div>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 bg-slate-950/90 backdrop-blur border border-slate-800 rounded-lg p-1 text-xs text-slate-300">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(0.6, z - 0.2))}
              className="p-1 hover:bg-slate-800 rounded text-slate-300 transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono px-1.5 text-slate-400">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.2))}
              className="p-1 hover:bg-slate-800 rounded text-slate-300 transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={resetView}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 transition border-l border-slate-800 ml-0.5 pl-1.5"
              title="Reset View"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Bottom Opacity Slider */}
      {showOverlay && gradcamBase64 && (
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-2 text-slate-400 text-[11px]">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>Grad-CAM Activation Opacity:</span>
          </div>

          <div className="flex items-center gap-3 flex-1 max-w-xs">
            <input
              type="range"
              min="10"
              max="100"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded cursor-pointer"
            />
            <span className="font-mono text-[11px] text-slate-300 w-8 text-right font-medium">
              {opacity}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
