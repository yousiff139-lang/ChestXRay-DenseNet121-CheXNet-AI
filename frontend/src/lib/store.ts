import { create } from "zustand";
import { ClassificationResponse, ModelComparisonResponse, UnifiedDiagnosisResponse } from "./types";

interface ChestXRayState {
  currentResult: ClassificationResponse | null;
  comparisonResult: ModelComparisonResponse | null;
  unifiedResult: UnifiedDiagnosisResponse | null;
  selectedModel: "densenet" | "resnet" | "both";
  showBoundingBoxes: boolean;
  showGradcam: boolean;
  gradcamOpacity: number;
  activeTab: "overview" | "detections" | "comparison" | "report" | "treatment";

  setCurrentResult: (result: ClassificationResponse | null) => void;
  setComparisonResult: (result: ModelComparisonResponse | null) => void;
  setUnifiedResult: (result: UnifiedDiagnosisResponse | null) => void;
  setSelectedModel: (model: "densenet" | "resnet" | "both") => void;
  setShowBoundingBoxes: (show: boolean) => void;
  setShowGradcam: (show: boolean) => void;
  setGradcamOpacity: (opacity: number) => void;
  setActiveTab: (tab: "overview" | "detections" | "comparison" | "report" | "treatment") => void;
  reset: () => void;
}

export const useChestXRayStore = create<ChestXRayState>((set) => ({
  currentResult: null,
  comparisonResult: null,
  unifiedResult: null,
  selectedModel: "densenet",
  showBoundingBoxes: true,
  showGradcam: true,
  gradcamOpacity: 50,
  activeTab: "overview",

  setCurrentResult: (result) => set({ currentResult: result }),
  setComparisonResult: (result) => set({ comparisonResult: result }),
  setUnifiedResult: (result) => set({ unifiedResult: result }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  setShowBoundingBoxes: (show) => set({ showBoundingBoxes: show }),
  setShowGradcam: (show) => set({ showGradcam: show }),
  setGradcamOpacity: (opacity) => set({ gradcamOpacity: opacity }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  reset: () =>
    set({
      currentResult: null,
      comparisonResult: null,
      unifiedResult: null,
    }),
}));
