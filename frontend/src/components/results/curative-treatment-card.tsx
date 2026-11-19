"use client";

import React from "react";
import { CurativeProtocol } from "../../lib/types";
import { CheckCircle2, Clock, Pill, Stethoscope } from "lucide-react";

interface CurativeTreatmentCardProps {
  protocols: CurativeProtocol[];
}

export const CurativeTreatmentCard: React.FC<CurativeTreatmentCardProps> = ({ protocols }) => {
  if (!protocols || protocols.length === 0) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
        <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
        <h4 className="font-semibold text-slate-200 text-sm">No Acute Pathologies Detected</h4>
        <p className="text-xs text-slate-400 mt-1">
          The radiograph does not demonstrate acute findings requiring immediate therapeutic intervention.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-slate-100 text-sm">
          Treatment &amp; Clinical Guidelines ({protocols.length})
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {protocols.map((protocol, idx) => (
          <div
            key={idx}
            className="bg-slate-900/80 border border-slate-800 hover:border-slate-700 rounded-xl p-4 flex flex-col justify-between transition"
          >
            <div>
              {/* Header */}
              <div className="flex items-center gap-2 border-b border-slate-800 pb-2.5 mb-3">
                <span className="w-2 h-2 rounded-full bg-blue-400" />
                <span className="font-semibold text-slate-100 text-sm">
                  {protocol.pathology}
                </span>
              </div>

              {/* Primary Action */}
              <div className="mb-3">
                <span className="text-[11px] font-medium text-slate-400 block mb-1">
                  Clinical Action
                </span>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {protocol.clinical_action}
                </p>
              </div>

              {/* Medications */}
              {protocol.recommended_medications && protocol.recommended_medications.length > 0 && (
                <div className="mb-3">
                  <span className="text-[11px] font-medium text-amber-300 flex items-center gap-1 mb-1">
                    <Pill className="w-3 h-3" />
                    Medications
                  </span>
                  <ul className="space-y-1">
                    {protocol.recommended_medications.map((med, mIdx) => (
                      <li
                        key={mIdx}
                        className="text-xs text-slate-300 bg-slate-950/70 border border-slate-800/80 rounded px-2.5 py-1"
                      >
                        {med}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Procedural Intervention */}
              {protocol.procedural_intervention && (
                <div className="mb-3">
                  <span className="text-[11px] font-medium text-blue-300 flex items-center gap-1 mb-1">
                    <Stethoscope className="w-3 h-3" />
                    Procedure / Investigation
                  </span>
                  <p className="text-xs text-slate-300 bg-slate-950/70 border border-slate-800/80 rounded p-2">
                    {protocol.procedural_intervention}
                  </p>
                </div>
              )}
            </div>

            {/* Monitoring & Follow-Up */}
            <div className="border-t border-slate-800 pt-2.5 mt-2 flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <span>Follow-up:</span>
              </span>
              <span className="font-medium text-slate-200">
                {protocol.follow_up_timeline}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
