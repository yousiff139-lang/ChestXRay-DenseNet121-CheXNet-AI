"use client";

import React from "react";
import { CurativeProtocol } from "../../lib/types";
import { Activity, Pill, Stethoscope, Clock, CheckCircle2 } from "lucide-react";

interface CurativeTreatmentCardProps {
  protocols: CurativeProtocol[];
}

export const CurativeTreatmentCard: React.FC<CurativeTreatmentCardProps> = ({ protocols }) => {
  if (!protocols || protocols.length === 0) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 text-center text-slate-400">
        <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-2" />
        <h4 className="font-semibold text-slate-200">No Acute Pathologies Requiring Intervention</h4>
        <p className="text-xs text-slate-400 mt-1">
          The radiograph does not demonstrate acute lesions requiring immediate therapeutic regimens.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h3 className="font-bold text-slate-100 text-base">
            Actionable Clinical & Curative Protocols
          </h3>
        </div>
        <span className="text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2.5 py-0.5 rounded-full font-mono">
          {protocols.length} Regimen(s)
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {protocols.map((protocol, idx) => (
          <div
            key={idx}
            className="bg-slate-900/80 border border-slate-800 hover:border-slate-700/80 rounded-2xl p-5 flex flex-col justify-between transition-all shadow-lg"
          >
            <div>
              {/* Header Badge */}
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-3">
                <span className="font-bold text-slate-100 text-sm flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  {protocol.pathology}
                </span>
                <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded">
                  Clinical Protocol
                </span>
              </div>

              {/* Primary Action */}
              <div className="mb-3">
                <span className="text-[11px] uppercase font-bold text-slate-400 tracking-wider block mb-1">
                  Primary Clinical Directive
                </span>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {protocol.clinical_action}
                </p>
              </div>

              {/* Medications */}
              {protocol.recommended_medications && protocol.recommended_medications.length > 0 && (
                <div className="mb-3">
                  <span className="text-[11px] uppercase font-bold text-amber-400 tracking-wider flex items-center gap-1.5 mb-1.5">
                    <Pill className="w-3 h-3" />
                    Pharmacological Therapy
                  </span>
                  <ul className="space-y-1">
                    {protocol.recommended_medications.map((med, mIdx) => (
                      <li
                        key={mIdx}
                        className="text-xs text-slate-300 bg-slate-850/60 border border-slate-800/60 rounded-md px-2.5 py-1"
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
                  <span className="text-[11px] uppercase font-bold text-purple-400 tracking-wider flex items-center gap-1.5 mb-1">
                    <Stethoscope className="w-3 h-3" />
                    Procedural & Diagnostic Interventions
                  </span>
                  <p className="text-xs text-slate-300 bg-purple-950/20 border border-purple-900/30 rounded-md p-2">
                    {protocol.procedural_intervention}
                  </p>
                </div>
              )}
            </div>

            {/* Monitoring & Follow-Up */}
            <div className="border-t border-slate-800/80 pt-3 mt-3 flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-cyan-400" />
                <span>Monitoring Interval:</span>
              </span>
              <span className="font-semibold text-slate-200 text-right">
                {protocol.follow_up_timeline}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
