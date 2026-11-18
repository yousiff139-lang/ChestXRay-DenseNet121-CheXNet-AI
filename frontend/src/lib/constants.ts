// Application constants for Chest X-Ray AI

export const FILE_SIZE_LIMITS = {
  MAX_SIZE: 15 * 1024 * 1024, // 15 MB
} as const;

export const SUPPORTED_FILE_EXTENSIONS = {
  "application/dicom": [".dcm", ".dicom"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
} as const;

export const SUPPORTED_DICOM_EXTENSIONS = SUPPORTED_FILE_EXTENSIONS;

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

export const CHEST_XRAY_CLASSES = [
  "Atelectasis",
  "Cardiomegaly",
  "Consolidation",
  "Edema",
  "Effusion",
  "Emphysema",
  "Fibrosis",
  "Hernia",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pleural_Thickening",
  "Pneumonia",
  "Pneumothorax",
] as const;

export const DISEASE_INFO: Record<
  string,
  {
    description: string;
    clinicalConcern: "High" | "Moderate" | "Low";
    badgeColor: string;
  }
> = {
  Atelectasis: {
    description: "Partial or complete collapse of the lung or lung lobe.",
    clinicalConcern: "Moderate",
    badgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  },
  Cardiomegaly: {
    description: "Enlarged heart with cardiothoracic ratio exceeding 0.50.",
    clinicalConcern: "High",
    badgeColor: "bg-red-500/20 text-red-300 border-red-500/30",
  },
  Consolidation: {
    description: "Alveolar spaces filled with fluid/exudate replacing air.",
    clinicalConcern: "High",
    badgeColor: "bg-rose-500/20 text-rose-300 border-rose-500/30",
  },
  Edema: {
    description: "Accumulation of fluid in pulmonary interstitium/alveoli.",
    clinicalConcern: "High",
    badgeColor: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  },
  Effusion: {
    description: "Fluid collection in the pleural space blunting costophrenic angles.",
    clinicalConcern: "High",
    badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
  },
  Emphysema: {
    description: "Hyperinflation with flattened diaphragms and loss of lung parenchyma.",
    clinicalConcern: "Moderate",
    badgeColor: "bg-orange-500/20 text-orange-300 border-orange-500/30",
  },
  Fibrosis: {
    description: "Reticular lung opacities representing interstitial scarring.",
    clinicalConcern: "Moderate",
    badgeColor: "bg-violet-500/20 text-violet-300 border-violet-500/30",
  },
  Hernia: {
    description: "Diaphragmatic herniation of abdominal contents into the thorax.",
    clinicalConcern: "High",
    badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/30",
  },
  Infiltration: {
    description: "Ill-defined parenchymal opacification from inflammation or fluid.",
    clinicalConcern: "Moderate",
    badgeColor: "bg-yellow-500/20 text-yellow-300 border-yellow-500/30",
  },
  Mass: {
    description: "Solitary pulmonary lesion greater than 3 cm in diameter.",
    clinicalConcern: "High",
    badgeColor: "bg-red-600/20 text-red-400 border-red-600/30",
  },
  Nodule: {
    description: "Discrete rounded pulmonary opacity measuring up to 3 cm.",
    clinicalConcern: "Moderate",
    badgeColor: "bg-amber-600/20 text-amber-400 border-amber-600/30",
  },
  Pleural_Thickening: {
    description: "Fibrotic or calcified pleural margin thickening.",
    clinicalConcern: "Low",
    badgeColor: "bg-slate-500/20 text-slate-300 border-slate-500/30",
  },
  Pneumonia: {
    description: "Infectious consolidation or patchy parenchymal opacities.",
    clinicalConcern: "High",
    badgeColor: "bg-rose-600/20 text-rose-300 border-rose-600/30",
  },
  Pneumothorax: {
    description: "Free air in the pleural cavity causing partial lung collapse.",
    clinicalConcern: "High",
    badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  },
};

export function getDiseaseSeverityColor(probability: number): string {
  if (probability >= 0.7) return "text-red-400 bg-red-500/10 border-red-500/30";
  if (probability >= 0.5) return "text-amber-400 bg-amber-500/10 border-amber-500/30";
  if (probability >= 0.25) return "text-blue-400 bg-blue-500/10 border-blue-500/30";
  return "text-slate-400 bg-slate-500/10 border-slate-500/20";
}
