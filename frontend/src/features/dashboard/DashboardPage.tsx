import React from 'react';
import { Link } from 'react-router-dom';
import {
  MessageSquareHeart,
  Activity,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  Compass,
  Clock,
} from 'lucide-react';
import { useConsent } from '../../context/ConsentContext';
import { WELLNESS_RESOURCES } from '../../services/wellnessResources';

export const DashboardPage: React.FC = () => {
  const { conversationStorage, analytics } = useConsent();

  const featuredResources = WELLNESS_RESOURCES.slice(0, 3);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-wellness-800 to-wellness-700 rounded-3xl p-6 sm:p-8 text-white shadow-lg relative overflow-hidden">
        <div className="relative z-10 max-w-2xl space-y-3">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-wellness-600/60 text-wellness-200 text-xs font-semibold backdrop-blur-sm">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Welcome to your calm space</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            How are you feeling right now?
          </h1>
          <p className="text-sm text-wellness-100 leading-relaxed">
            MindBridge is here to listen and help you unpack your thoughts without judgment. Share your thoughts or explore guided exercises at your own pace.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link
              to="/chat"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white text-wellness-800 font-semibold text-sm hover:bg-wellness-50 transition-colors shadow-sm"
            >
              <MessageSquareHeart className="w-4 h-4 text-wellness-600" />
              Start Wellness Chat
            </Link>
            <Link
              to="/resources"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-wellness-900/50 hover:bg-wellness-900/80 text-white font-medium text-sm transition-colors border border-wellness-600"
            >
              <Compass className="w-4 h-4 text-wellness-300" />
              Browse Coping Tools
            </Link>
          </div>
        </div>

        {/* Decorative circle */}
        <div className="absolute right-[-40px] bottom-[-40px] w-64 h-64 bg-wellness-600/30 rounded-full blur-2xl pointer-events-none" />
      </div>

      {/* Main Grid: Action Modules & Privacy */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Module 1: AI Chat Quick Launch */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm flex flex-col justify-between hover:border-wellness-300 transition-colors">
          <div>
            <div className="w-10 h-10 rounded-xl bg-wellness-50 text-wellness-600 flex items-center justify-center mb-4">
              <MessageSquareHeart className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-bold text-calm-900 mb-1">Empathetic Peer Chat</h2>
            <p className="text-xs sm:text-sm text-calm-600 leading-relaxed">
              Express your feelings, academic stress, or worries. Our multi-label models identify 28 emotion dimensions to give supportive reflections.
            </p>
          </div>
          <div className="pt-6">
            <Link
              to="/chat"
              className="w-full inline-flex items-center justify-between px-4 py-2.5 rounded-xl bg-wellness-50 hover:bg-wellness-100 text-wellness-800 font-medium text-xs transition-colors"
            >
              <span>Open Private Chat</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        {/* Module 2: Analytics & Reflection */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm flex flex-col justify-between hover:border-wellness-300 transition-colors">
          <div>
            <div className="w-10 h-10 rounded-xl bg-wellness-50 text-wellness-600 flex items-center justify-center mb-4">
              <Activity className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-bold text-calm-900 mb-1">Wellness Analytics</h2>
            <p className="text-xs sm:text-sm text-calm-600 leading-relaxed">
              View your privacy-conscious emotion trends, stress patterns, and session coping summaries.
            </p>
          </div>
          <div className="pt-6">
            <Link
              to="/analytics"
              className="w-full inline-flex items-center justify-between px-4 py-2.5 rounded-xl bg-wellness-50 hover:bg-wellness-100 text-wellness-800 font-medium text-xs transition-colors"
            >
              <span>View Analytics</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        {/* Module 3: Privacy & Consent Status */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm flex flex-col justify-between hover:border-wellness-300 transition-colors">
          <div>
            <div className="w-10 h-10 rounded-xl bg-sage-100 text-sage-700 flex items-center justify-center mb-4">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-bold text-calm-900 mb-1">Privacy & Consent Controls</h2>
            <p className="text-xs sm:text-sm text-calm-600 leading-relaxed">
              You own your data. Check your current consent permissions and privacy audit status.
            </p>
            <div className="mt-3 space-y-1.5 text-xs">
              <div className="flex items-center justify-between py-1 border-b border-calm-100">
                <span className="text-calm-500">Conversation Storage:</span>
                <span className={`font-semibold ${conversationStorage?.granted ? 'text-wellness-700' : 'text-calm-600'}`}>
                  {conversationStorage?.granted ? 'Granted' : 'In-Memory Only'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-calm-500">Analytics Sharing:</span>
                <span className={`font-semibold ${analytics?.granted ? 'text-wellness-700' : 'text-calm-600'}`}>
                  {analytics?.granted ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          </div>
          <div className="pt-4">
            <Link
              to="/consent"
              className="w-full inline-flex items-center justify-between px-4 py-2.5 rounded-xl bg-sage-50 hover:bg-sage-100 text-sage-800 font-medium text-xs transition-colors"
            >
              <span>Manage Consent Settings</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Recommended Self-Care & Coping Strategies */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-calm-900">Featured Wellness Strategies</h2>
            <p className="text-xs text-calm-500">Evidence-based exercises to help restore calm and balance</p>
          </div>
          <Link
            to="/resources"
            className="text-xs font-semibold text-wellness-700 hover:text-wellness-800 flex items-center gap-1"
          >
            Browse all ({WELLNESS_RESOURCES.length})
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {featuredResources.map((res) => (
            <div
              key={res.id}
              className="bg-white p-5 rounded-2xl border border-calm-200 shadow-sm flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between text-xs text-calm-400 mb-2">
                  <span className="font-semibold text-wellness-700 uppercase tracking-wider text-[10px] px-2 py-0.5 bg-wellness-50 rounded">
                    {res.category.replace('_', ' ')}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {res.readTimeMinutes} min read
                  </span>
                </div>
                <h3 className="font-semibold text-sm text-calm-900 mb-1.5">{res.title}</h3>
                <p className="text-xs text-calm-600 leading-relaxed line-clamp-2">{res.summary}</p>
              </div>
              <div className="pt-4 mt-4 border-t border-calm-100 flex items-center justify-between">
                <Link
                  to={`/resources#${res.id}`}
                  className="text-xs font-medium text-wellness-700 hover:text-wellness-800 flex items-center gap-1"
                >
                  View steps
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
