import React from 'react';
import { PhoneCall, ShieldCheck, Heart } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-calm-200 mt-auto">
      {/* Crisis Lifeline Ribbon */}
      <div className="bg-wellness-900 text-white py-3 px-4">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs sm:text-sm">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="font-semibold text-wellness-200">24/7 Crisis Support:</span>
            <span>If you are in danger or need crisis intervention:</span>
          </div>
          <div className="flex flex-wrap items-center gap-3 font-medium text-xs">
            <a
              href="tel:14416"
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-wellness-800 hover:bg-wellness-700 transition-colors text-wellness-200 hover:text-white"
              title="India National Helpline"
            >
              <PhoneCall className="w-3 h-3 text-emerald-300" />
              India: 14416 (Tele-MANAS)
            </a>
            <span className="text-wellness-500">|</span>
            <a
              href="tel:988"
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-wellness-700 hover:bg-wellness-600 transition-colors text-white"
              title="US & Canada"
            >
              <PhoneCall className="w-3 h-3 text-wellness-300" />
              US/CA: 988
            </a>
            <span className="text-wellness-500">|</span>
            <a
              href="sms:741741"
              className="text-wellness-200 hover:text-white underline underline-offset-2"
            >
              SMS: 741741
            </a>
            <span className="text-wellness-500">|</span>
            <Link
              to="/resources"
              className="text-wellness-300 hover:text-white underline underline-offset-2"
            >
              All Helplines
            </Link>
          </div>
        </div>
      </div>

      {/* Main Footer Details & Clinical Disclaimers */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <h3 className="font-semibold text-calm-900 text-base flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-wellness-600" />
              MindBridge AI Mental Wellness & Peer Support
            </h3>
            <p className="mt-2 text-xs sm:text-sm text-calm-600 leading-relaxed">
              MindBridge is an autonomous AI-powered peer wellness companion designed to analyze emotional states, suggest coping strategies, and connect users with verified resources.
            </p>
            <div className="mt-3 p-3 bg-calm-50 rounded-lg border border-calm-200 text-xs text-calm-600 leading-relaxed">
              <span className="font-semibold text-calm-800">Critical Medical & Safety Disclaimer: </span>
              MindBridge is not a medical device, licensed medical clinic, psychiatrist, or psychological therapy service. The platform does not diagnose medical or mental health illnesses, prescribe treatments, or substitute for professional medical advice. Conversations are processed in-memory under strict privacy principles.
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-calm-900 text-sm mb-3">Quick Navigation</h4>
            <ul className="space-y-2 text-xs text-calm-600">
              <li><Link to="/chat" className="hover:text-wellness-600 transition-colors">AI Wellness Chat</Link></li>
              <li><Link to="/resources" className="hover:text-wellness-600 transition-colors">Coping Strategies & Guides</Link></li>
              <li><Link to="/analytics" className="hover:text-wellness-600 transition-colors">Personal Wellness Trends</Link></li>
              <li><Link to="/consent" className="hover:text-wellness-600 transition-colors">Privacy & Data Controls</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-calm-900 text-sm mb-3">Trust & Security</h4>
            <ul className="space-y-2 text-xs text-calm-600">
              <li><span className="text-calm-700 font-medium">Zero-Message Server Persistence:</span> Messages are evaluated in-memory.</li>
              <li><span className="text-calm-700 font-medium">Granular Consent:</span> You control research and analytics sharing.</li>
              <li><span className="text-calm-700 font-medium">Audit-Logged Security:</span> Tamper-evident privacy record keeping.</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-calm-200 flex flex-col sm:flex-row items-center justify-between text-xs text-calm-500 gap-2">
          <p>© {new Date().getFullYear()} MindBridge Peer Support. All rights reserved.</p>
          <p className="flex items-center gap-1">
            Engineered with <Heart className="w-3.5 h-3.5 text-wellness-600 fill-wellness-600" /> for empathetic, ethical student wellness.
          </p>
        </div>
      </div>
    </footer>
  );
};
