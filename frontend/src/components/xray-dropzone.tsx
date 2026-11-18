"use client";

import React, { useRef, useState } from "react";
import { Upload, FileText, Image as ImageIcon, AlertCircle } from "lucide-react";
import { formatFileSize, FILE_SIZE_LIMITS } from "../lib/constants";

interface XRayDropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const XRayDropzone: React.FC<XRayDropzoneProps> = ({
  onFileSelect,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSelect = (file: File) => {
    setError(null);
    if (file.size > FILE_SIZE_LIMITS.MAX_SIZE) {
      setError(`File size exceeds ${formatFileSize(FILE_SIZE_LIMITS.MAX_SIZE)} limit.`);
      return;
    }

    const validExtensions = [".png", ".jpg", ".jpeg", ".dcm", ".dicom"];
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!validExtensions.includes(ext)) {
      setError("Please upload a DICOM (.dcm), PNG, or JPEG chest X-ray image.");
      return;
    }

    onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelect(e.target.files[0]);
    }
  };

  const loadSample = async (samplePath: string, fileName: string) => {
    try {
      setLoadingSample(fileName);
      setError(null);
      const res = await fetch(samplePath);
      const blob = await res.blob();
      const file = new File([blob], fileName, { type: blob.type || "image/jpeg" });
      onFileSelect(file);
    } catch {
      setError("Failed to load sample image. Please upload a local file.");
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Upload Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`relative cursor-pointer border rounded-xl p-8 transition-colors flex flex-col items-center justify-center text-center ${
          isDragging
            ? "border-blue-500 bg-blue-500/5"
            : disabled
            ? "border-slate-800 bg-slate-900/40 opacity-60 cursor-not-allowed"
            : "border-slate-800 hover:border-slate-700 bg-slate-900/60 hover:bg-slate-900/90"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.dcm,.dicom"
          className="hidden"
          disabled={disabled}
          onChange={handleInputChange}
        />

        <div className="w-12 h-12 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center mb-3 text-slate-300">
          <Upload className="w-6 h-6 text-blue-400" />
        </div>

        <h3 className="text-sm font-semibold text-slate-200 mb-1">
          Select or Drop Frontal Chest Radiograph
        </h3>
        <p className="text-xs text-slate-400 max-w-sm mb-4">
          Upload DICOM or standard image formats for multi-label thoracic disease inference & Grad-CAM analysis
        </p>

        <div className="flex flex-wrap items-center justify-center gap-2 text-[11px] text-slate-400">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60">
            <ImageIcon className="w-3 h-3 text-blue-400" /> PNG / JPEG
          </span>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60">
            <FileText className="w-3 h-3 text-emerald-400" /> DICOM (.dcm)
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 text-slate-400">
            Max 15MB
          </span>
        </div>
      </div>

      {/* Quick Test Samples */}
      <div className="p-3.5 rounded-xl bg-slate-900/40 border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Verified Clinical Test Cohort (Real Patients)
          </span>
          <span className="text-[10px] text-slate-500">Board-Certified Ground Truth</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <button
            type="button"
            disabled={disabled || !!loadingSample}
            onClick={() => loadSample("/samples/real_normal_case.jpg", "real_patient_normal.jpg")}
            className="px-2.5 py-2 text-left rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-emerald-500/30 text-xs transition disabled:opacity-50"
          >
            <div className="font-medium text-emerald-400 text-[11px]">Real Normal</div>
            <div className="text-[10px] text-slate-400 truncate">Healthy Lungs Control</div>
          </button>
          <button
            type="button"
            disabled={disabled || !!loadingSample}
            onClick={() => loadSample("/samples/real_pneumonia_case.jpg", "real_patient_pneumonia.jpg")}
            className="px-2.5 py-2 text-left rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-rose-500/30 text-xs transition disabled:opacity-50"
          >
            <div className="font-medium text-rose-300 text-[11px]">Viral Pneumonia</div>
            <div className="text-[10px] text-slate-400 truncate">Bilateral Infiltrates</div>
          </button>
          <button
            type="button"
            disabled={disabled || !!loadingSample}
            onClick={() => loadSample("/samples/real_bacterial_pneumonia.jpg", "real_patient_bacterial.jpg")}
            className="px-2.5 py-2 text-left rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-amber-500/30 text-xs transition disabled:opacity-50"
          >
            <div className="font-medium text-amber-300 text-[11px]">Bacterial Pneumonia</div>
            <div className="text-[10px] text-slate-400 truncate">Streptococcus Opacity</div>
          </button>
          <button
            type="button"
            disabled={disabled || !!loadingSample}
            onClick={() => loadSample("/samples/real_severe_ards.jpg", "real_patient_severe_ards.jpg")}
            className="px-2.5 py-2 text-left rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-blue-500/30 text-xs transition disabled:opacity-50"
          >
            <div className="font-medium text-blue-300 text-[11px]">Severe ARDS</div>
            <div className="text-[10px] text-slate-400 truncate">Alveolar Infiltrates</div>
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
