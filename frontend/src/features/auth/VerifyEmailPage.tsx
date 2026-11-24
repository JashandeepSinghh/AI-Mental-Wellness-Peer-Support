import React, { useState, useEffect } from 'react';
import { useSearchParams, Link, useLocation } from 'react-router-dom';
import { Mail, CheckCircle2, AlertCircle, RefreshCw, ArrowRight, Loader2 } from 'lucide-react';
import { api } from '../../services/api';

export const VerifyEmailPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const tokenFromUrl = searchParams.get('token');

  const [email, setEmail] = useState<string>((location.state as any)?.email || '');
  const [verificationStatus, setVerificationStatus] = useState<'idle' | 'verifying' | 'success' | 'error'>(
    tokenFromUrl ? 'verifying' : 'idle'
  );
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [resendLoading, setResendLoading] = useState(false);
  const [resendFeedback, setResendFeedback] = useState<string | null>(null);

  useEffect(() => {
    if (tokenFromUrl) {
      const verify = async () => {
        setVerificationStatus('verifying');
        try {
          const res = await api.verifyEmail(tokenFromUrl);
          if (res.success) {
            setVerificationStatus('success');
            setStatusMessage(res.message || 'Email verified successfully! You can now log in.');
          } else {
            setVerificationStatus('error');
            setStatusMessage(res.message || 'Verification link is invalid or expired.');
          }
        } catch (err: any) {
          setVerificationStatus('error');
          setStatusMessage(err.message || 'Verification failed. The link may have expired or was already used.');
        }
      };
      verify();
    }
  }, [tokenFromUrl]);

  const handleResend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;

    setResendLoading(true);
    setResendFeedback(null);
    try {
      const res = await api.resendVerification(email.trim());
      setResendFeedback(res.message || 'If this account is eligible, a new verification link has been sent.');
    } catch (err: any) {
      setResendFeedback(err.message || 'Failed to resend verification. Please try again.');
    } finally {
      setResendLoading(false);
    }
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-white p-8 sm:p-10 rounded-3xl border border-calm-200 shadow-xl shadow-calm-100 text-center">
        {/* State: In-progress verification from URL */}
        {verificationStatus === 'verifying' && (
          <div className="space-y-4">
            <Loader2 className="w-12 h-12 text-wellness-600 animate-spin mx-auto" />
            <h2 className="text-2xl font-bold text-calm-900">Verifying Email...</h2>
            <p className="text-sm text-calm-500">Checking your security token with the server.</p>
          </div>
        )}

        {/* State: Verification Successful */}
        {verificationStatus === 'success' && (
          <div className="space-y-5">
            <div className="w-14 h-14 rounded-2xl bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <h2 className="text-2xl font-bold text-calm-900">Email Verified!</h2>
            <p className="text-sm text-calm-600 leading-relaxed">{statusMessage}</p>
            <Link
              to="/login"
              className="inline-flex items-center justify-center gap-2 w-full py-3 px-4 rounded-xl bg-wellness-600 hover:bg-wellness-700 text-white font-medium text-sm shadow-md transition-all"
            >
              Proceed to Sign In
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        )}

        {/* State: Verification Failed or Token Expired */}
        {verificationStatus === 'error' && (
          <div className="space-y-5">
            <div className="w-14 h-14 rounded-2xl bg-red-100 text-red-600 flex items-center justify-center mx-auto">
              <AlertCircle className="w-8 h-8" />
            </div>
            <h2 className="text-2xl font-bold text-calm-900">Verification Link Issue</h2>
            <p className="text-sm text-red-600 leading-relaxed">{statusMessage}</p>
            <p className="text-xs text-calm-500">
              Verification links expire for your safety. Request a new one below:
            </p>
          </div>
        )}

        {/* State: General Instructions or Resend Form */}
        {(verificationStatus === 'idle' || verificationStatus === 'error') && (
          <div className="space-y-6 text-left">
            {verificationStatus === 'idle' && (
              <div className="text-center">
                <div className="w-12 h-12 rounded-2xl bg-wellness-100 text-wellness-700 flex items-center justify-center mx-auto mb-3">
                  <Mail className="w-6 h-6" />
                </div>
                <h2 className="text-2xl font-bold text-calm-900">Check Your Inbox</h2>
                <p className="mt-2 text-xs sm:text-sm text-calm-500 leading-relaxed">
                  We've dispatched a confidential verification link. Please click the link in your email to activate your account.
                </p>
              </div>
            )}

            <div className="p-4 bg-calm-50 rounded-2xl border border-calm-200">
              <h3 className="text-xs font-semibold text-calm-800 uppercase tracking-wider mb-2">
                Didn't receive the email?
              </h3>
              <form onSubmit={handleResend} className="space-y-3">
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your registered email"
                  className="w-full px-3.5 py-2 rounded-xl border border-calm-200 focus:outline-none focus:ring-2 focus:ring-wellness-500 text-xs sm:text-sm text-calm-900"
                />
                <button
                  type="submit"
                  disabled={resendLoading || !email.trim()}
                  className="w-full py-2.5 px-3 rounded-xl bg-wellness-700 hover:bg-wellness-800 text-white font-medium text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                >
                  {resendLoading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Sending...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-3.5 h-3.5" />
                      Resend Verification Link
                    </>
                  )}
                </button>
              </form>
              {resendFeedback && (
                <p className="mt-2.5 text-xs text-wellness-700 bg-wellness-50 p-2.5 rounded-lg border border-wellness-100">
                  {resendFeedback}
                </p>
              )}
            </div>

            <div className="text-center pt-2">
              <Link to="/login" className="text-xs font-semibold text-wellness-700 hover:text-wellness-800">
                Back to Sign In
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
