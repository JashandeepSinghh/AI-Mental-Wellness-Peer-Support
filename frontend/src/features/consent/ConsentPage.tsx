import React, { useState } from 'react';
import {
  ShieldCheck,
  Database,
  BarChart3,
  AlertCircle,
  CheckCircle2,
  Clock,
  FileText,
} from 'lucide-react';
import { useConsent } from '../../context/ConsentContext';

export const ConsentPage: React.FC = () => {
  const { conversationStorage, analytics, isLoading, error, updateConsent } = useConsent();
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const handleToggle = async (purpose: 'conversation_storage' | 'analytics', currentValue: boolean) => {
    setSuccessMsg(null);
    try {
      await updateConsent(purpose, !currentValue);
      setSuccessMsg(
        `Successfully ${!currentValue ? 'granted' : 'withdrawn'} consent for ${
          purpose === 'conversation_storage' ? 'conversation storage' : 'wellness analytics'
        }.`
      );
    } catch {
      // Error handled by context
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-wellness-100 text-wellness-800 text-xs font-semibold">
          <ShieldCheck className="w-3.5 h-3.5 text-wellness-600" />
          <span>Privacy & Data Sovereignty</span>
        </div>
        <h1 className="text-3xl font-bold text-calm-950 tracking-tight">Consent Management</h1>
        <p className="text-sm text-calm-600 max-w-2xl leading-relaxed">
          At MindBridge, student privacy is a foundational pillar. You have complete authority over how your data is processed, stored, or analyzed. Every decision is recorded into an append-only, tamper-evident audit log.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3 text-red-700 text-xs sm:text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-red-600" />
          <p>{error}</p>
        </div>
      )}

      {successMsg && (
        <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 flex items-start gap-3 text-emerald-800 text-xs sm:text-sm">
          <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-600" />
          <p>{successMsg}</p>
        </div>
      )}

      {/* Consent Toggles */}
      <div className="space-y-6">
        {/* Purpose 1: Conversation Storage */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-start gap-3.5">
              <div className="w-10 h-10 rounded-xl bg-wellness-50 text-wellness-700 flex items-center justify-center shrink-0 mt-0.5">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-calm-900">Conversation Storage</h2>
                <p className="text-xs sm:text-sm text-calm-600 leading-relaxed mt-1">
                  Permits persisting encrypted chat messages to allow past session retrieval. When denied or withdrawn, all chat interactions are strictly ephemeral and processed in-memory only.
                </p>
              </div>
            </div>

            <button
              onClick={() => handleToggle('conversation_storage', conversationStorage?.granted ?? false)}
              disabled={isLoading}
              className={`shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-wellness-500 focus:ring-offset-2 ${
                conversationStorage?.granted ? 'bg-wellness-600' : 'bg-calm-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  conversationStorage?.granted ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <div className="pt-3 border-t border-calm-100 flex flex-wrap items-center justify-between text-xs text-calm-500 gap-2">
            <span className="flex items-center gap-1.5 font-medium">
              Status:{' '}
              <span
                className={`font-semibold ${
                  conversationStorage?.granted ? 'text-wellness-700' : 'text-calm-700'
                }`}
              >
                {conversationStorage?.granted ? 'Granted (Encrypted Storage Active)' : 'Denied (In-Memory Processing Only)'}
              </span>
            </span>
            {conversationStorage?.recorded_at && (
              <span className="flex items-center gap-1 text-[11px] text-calm-400">
                <Clock className="w-3 h-3" />
                Updated: {new Date(conversationStorage.recorded_at).toLocaleString()}
              </span>
            )}
          </div>
        </div>

        {/* Purpose 2: Analytics & Research Sharing */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-start gap-3.5">
              <div className="w-10 h-10 rounded-xl bg-wellness-50 text-wellness-700 flex items-center justify-center shrink-0 mt-0.5">
                <BarChart3 className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-calm-900">Aggregated Wellness Analytics</h2>
                <p className="text-xs sm:text-sm text-calm-600 leading-relaxed mt-1">
                  Permits computing anonymized, aggregated emotional trends and model inference accuracy statistics. Raw user messages are never exported or used in analytics.
                </p>
              </div>
            </div>

            <button
              onClick={() => handleToggle('analytics', analytics?.granted ?? false)}
              disabled={isLoading}
              className={`shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-wellness-500 focus:ring-offset-2 ${
                analytics?.granted ? 'bg-wellness-600' : 'bg-calm-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  analytics?.granted ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <div className="pt-3 border-t border-calm-100 flex flex-wrap items-center justify-between text-xs text-calm-500 gap-2">
            <span className="flex items-center gap-1.5 font-medium">
              Status:{' '}
              <span
                className={`font-semibold ${
                  analytics?.granted ? 'text-wellness-700' : 'text-calm-700'
                }`}
              >
                {analytics?.granted ? 'Granted (De-identified Analytics Enabled)' : 'Denied (Excluded from Analytics)'}
              </span>
            </span>
            {analytics?.recorded_at && (
              <span className="flex items-center gap-1 text-[11px] text-calm-400">
                <Clock className="w-3 h-3" />
                Updated: {new Date(analytics.recorded_at).toLocaleString()}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Privacy Guarantees & Policy Version */}
      <div className="bg-calm-50 p-6 rounded-2xl border border-calm-200 text-xs text-calm-600 space-y-3">
        <h3 className="font-bold text-calm-900 flex items-center gap-2 text-sm">
          <FileText className="w-4 h-4 text-wellness-600" />
          MindBridge Privacy Architecture Guarantee (Policy Version 1.0)
        </h3>
        <ul className="list-disc list-inside space-y-1.5 leading-relaxed">
          <li><strong>No Coercion:</strong> You can access all chat features and crisis resources regardless of consent decisions.</li>
          <li><strong>Right to Withdraw:</strong> You can withdraw consent at any time; withdrawals take effect immediately.</li>
          <li><strong>Append-Only Auditability:</strong> Consent modifications create permanent audit trail records with UTC timestamps.</li>
          <li><strong>No Third-Party Advertising:</strong> MindBridge never sells, monetizes, or shares user data with marketing platforms.</li>
        </ul>
      </div>
    </div>
  );
};
