"use client";

import React, { useRef, useEffect, useState } from "react";
import { ThoracicDetection } from "../lib/types";
import { Eye, EyeOff, Layers } from "lucide-react";

interface BoundingBoxCanvasProps {
  imageSrc: string;
  gradcamSrc?: string | null;
  detections: ThoracicDetection[];
  showBoxes?: boolean;
  showGradcam?: boolean;
  gradcamOpacity?: number; // 0 to 100
  onToggleBoxes?: (show: boolean) => void;
  onToggleGradcam?: (show: boolean) => void;
  onChangeOpacity?: (opacity: number) => void;
}

export const BoundingBoxCanvas: React.FC<BoundingBoxCanvasProps> = ({
  imageSrc,
  gradcamSrc,
  detections = [],
  showBoxes = true,
  showGradcam = true,
  gradcamOpacity = 50,
  onToggleBoxes,
  onToggleGradcam,
  onChangeOpacity,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedDetection, setSelectedDetection] = useState<ThoracicDetection | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const render = () => {
      const w = img.clientWidth;
      const h = img.clientHeight;

      canvas.width = w;
      canvas.height = h;
      ctx.clearRect(0, 0, w, h);

      if (!showBoxes || detections.length === 0) return;

      detections.forEach((d) => {
        // Calculate box in rendered canvas coordinates
        const x = d.bbox.rel_x * w;
        const y = d.bbox.rel_y * h;
        const bw = d.bbox.rel_width * w;
        const bh = d.bbox.rel_height * h;

        const isSelected = selectedDetection?.id === d.id;
        const color = d.color || "#ef4444";

        // Subtle transparent fill
        ctx.fillStyle = `${color}18`;
        ctx.fillRect(x, y, bw, bh);

        // Clean crisp bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(x, y, bw, bh);

        // Clean top label badge
        const text = `${d.class_name} ${Math.round(d.confidence * 100)}%`;
        ctx.font = "bold 11px Inter, system-ui, sans-serif";
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const badgeHeight = 18;
        const badgeWidth = textWidth + 12;

        const badgeY = Math.max(0, y - badgeHeight - 2);

        // Badge background
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, badgeY, badgeWidth, badgeHeight, 3);
        ctx.fill();

        // Badge text
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, x + 6, badgeY + 13);
      });
    };

    render();

    const resizeObserver = new ResizeObserver(() => render());
    resizeObserver.observe(img);

    return () => resizeObserver.disconnect();
  }, [detections, showBoxes, selectedDetection]);

  return (
    <div className="flex flex-col gap-2.5">
      {/* Viewer Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/90 border border-slate-800 px-3.5 py-2 rounded-xl">
        <div className="flex items-center gap-2">
          {/* Toggle Bounding Boxes */}
          <button
            type="button"
            onClick={() => onToggleBoxes?.(!showBoxes)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              showBoxes
                ? "bg-blue-600/20 text-blue-300 border border-blue-500/30"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            {showBoxes ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Bounding Boxes ({detections.length})</span>
          </button>

          {/* Toggle Grad-CAM Heatmap */}
          {gradcamSrc && (
            <button
              type="button"
              onClick={() => onToggleGradcam?.(!showGradcam)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                showGradcam
                  ? "bg-amber-600/20 text-amber-300 border border-amber-500/30"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Grad-CAM Heatmap</span>
            </button>
          )}
        </div>

        {/* Grad-CAM Opacity Slider */}
        {gradcamSrc && showGradcam && (
          <div className="flex items-center gap-2 text-xs text-slate-300">
            <span className="text-slate-400">Blend:</span>
            <input
              type="range"
              min={0}
              max={100}
              value={gradcamOpacity}
              onChange={(e) => onChangeOpacity?.(Number(e.target.value))}
              className="w-20 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <span className="text-slate-400 w-7 text-right">{gradcamOpacity}%</span>
          </div>
        )}
      </div>

      {/* Radiograph Viewport */}
      <div
        ref={containerRef}
        className="relative overflow-hidden rounded-xl bg-black border border-slate-800 flex items-center justify-center min-h-[420px]"
      >
        {/* Base X-Ray Image */}
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Chest Radiograph"
          className="w-full max-h-[640px] object-contain select-none block"
        />

        {/* Grad-CAM Heatmap Overlay */}
        {gradcamSrc && showGradcam && (
          <img
            src={
              gradcamSrc.startsWith("data:image")
                ? gradcamSrc
                : `data:image/png;base64,${gradcamSrc}`
            }
            alt="Grad-CAM Heatmap"
            style={{ opacity: gradcamOpacity / 100 }}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150 mix-blend-screen"
          />
        )}

        {/* HTML5 Canvas for Bounding Boxes */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Bounding Box Legend */}
      {showBoxes && detections.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-[11px] font-medium text-slate-400">
            Detected:
          </span>
          {detections.map((d) => (
            <button
              key={d.id}
              type="button"
              onClick={() => setSelectedDetection(selectedDetection?.id === d.id ? null : d)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition ${
                selectedDetection?.id === d.id
                  ? "bg-slate-800 border-slate-400 text-white"
                  : "bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700"
              }`}
            >
              <span
                className="w-2 h-2 rounded-full inline-block"
                style={{ backgroundColor: d.color }}
              />
              <span>{d.class_name}</span>
              <span className="text-slate-400 text-[11px]">
                {Math.round(d.confidence * 100)}%
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
