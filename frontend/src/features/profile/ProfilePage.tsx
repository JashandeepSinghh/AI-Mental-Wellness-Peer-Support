import React from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { User, ShieldCheck, LogOut, Lock, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useConsent } from '../../context/ConsentContext';

export const ProfilePage: React.FC = () => {
  const { user, logout } = useAuth();
  const { conversationStorage, analytics } = useConsent();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-wellness-100 text-wellness-800 text-xs font-semibold">
          <User className="w-3.5 h-3.5 text-wellness-600" />
          <span>Account & Security Settings</span>
        </div>
        <h1 className="text-3xl font-bold text-calm-950 tracking-tight">Your Account Profile</h1>
        <p className="text-sm text-calm-600">
          Manage your authenticated session, verify cryptographic security credentials, and review your privacy settings.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column: Account Details */}
        <div className="md:col-span-2 space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-5">
            <h2 className="text-base font-bold text-calm-900 border-b border-calm-100 pb-3 flex items-center gap-2">
              <User className="w-4 h-4 text-wellness-600" />
              Account Credentials
            </h2>

            <div className="space-y-3 text-xs sm:text-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 border-b border-calm-100 gap-1">
                <span className="text-calm-500 font-medium">User Identifier (UUID):</span>
                <span className="font-mono text-xs bg-calm-100 px-2 py-1 rounded text-calm-800">
                  {user?.id || 'Anonymous Student'}
                </span>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 border-b border-calm-100 gap-1">
                <span className="text-calm-500 font-medium">Account Role:</span>
                <span className="capitalize font-semibold text-wellness-700 bg-wellness-50 px-2.5 py-0.5 rounded-full border border-wellness-200 text-xs">
                  {user?.role || 'student'}
                </span>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 border-b border-calm-100 gap-1">
                <span className="text-calm-500 font-medium">Authentication Type:</span>
                <span className="text-calm-700 font-medium">
                  Argon2 Hashed Password + JWT Bearer
                </span>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 gap-1">
                <span className="text-calm-500 font-medium">Refresh Token Security:</span>
                <span className="text-emerald-700 font-semibold flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  Rotated HttpOnly Cookie
                </span>
              </div>
            </div>
          </div>

          {/* Privacy Snapshot Card */}
          <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-calm-100 pb-3">
              <h2 className="text-base font-bold text-calm-900 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-wellness-600" />
                Active Consent Status
              </h2>
              <Link to="/consent" className="text-xs font-semibold text-wellness-700 hover:underline">
                Edit Preferences
              </Link>
            </div>

            <div className="space-y-3 text-xs sm:text-sm">
              <div className="flex items-center justify-between">
                <span className="text-calm-600">Conversation Storage:</span>
                <span
                  className={`font-semibold px-2 py-0.5 rounded text-xs ${
                    conversationStorage?.granted
                      ? 'bg-wellness-100 text-wellness-800'
                      : 'bg-calm-100 text-calm-700'
                  }`}
                >
                  {conversationStorage?.granted ? 'Granted' : 'In-Memory Only'}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-calm-600">Aggregated Analytics:</span>
                <span
                  className={`font-semibold px-2 py-0.5 rounded text-xs ${
                    analytics?.granted
                      ? 'bg-wellness-100 text-wellness-800'
                      : 'bg-calm-100 text-calm-700'
                  }`}
                >
                  {analytics?.granted ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Actions */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
            <h3 className="font-bold text-sm text-calm-900">Session Controls</h3>
            <p className="text-xs text-calm-500 leading-relaxed">
              Terminating your session will revoke your server refresh token and purge the local authentication context.
            </p>
            <button
              onClick={handleLogout}
              className="w-full py-2.5 px-4 rounded-xl border border-red-200 bg-red-50 hover:bg-red-100 text-red-700 font-medium text-xs flex items-center justify-center gap-2 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Sign Out from Device
            </button>
          </div>

          <div className="bg-wellness-50 p-6 rounded-2xl border border-wellness-200 space-y-3 text-xs text-wellness-900">
            <h4 className="font-bold flex items-center gap-1.5">
              <Lock className="w-4 h-4 text-wellness-700" />
              Zero-Trust Architecture
            </h4>
            <p className="leading-relaxed text-wellness-800">
              Your passwords are never stored in plaintext. Access tokens expire quickly (15 minutes), and refresh tokens are restricted to HttpOnly cookies, rendering them unreachable by cross-site scripting (XSS).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
