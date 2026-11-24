import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { HeartHandshake, Mail, Lock, AlertCircle, CheckCircle2, ArrowRight, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export const RegisterPage: React.FC = () => {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [termsAgreed, setTermsAgreed] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const validatePassword = (pwd: string) => {
    return {
      length: pwd.length >= 8,
      uppercase: /[A-Z]/.test(pwd),
      lowercase: /[a-z]/.test(pwd),
      number: /[0-9]/.test(pwd),
    };
  };

  const pwdChecks = validatePassword(password);
  const isPwdValid = Object.values(pwdChecks).every(Boolean);
  const doPasswordsMatch = password === confirmPassword && confirmPassword.length > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    if (!termsAgreed) {
      setErrorMessage('Please acknowledge the privacy & terms agreement to continue.');
      return;
    }

    if (!isPwdValid) {
      setErrorMessage('Please satisfy all password complexity criteria.');
      return;
    }

    if (!doPasswordsMatch) {
      setErrorMessage('Passwords do not match.');
      return;
    }

    setIsLoading(true);
    try {
      const res = await register(email, password);
      if (res.success) {
        navigate('/verify-email-instruction', { state: { email } });
      } else {
        setErrorMessage(res.message || 'Registration could not be completed.');
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred during registration.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-white p-8 sm:p-10 rounded-3xl border border-calm-200 shadow-xl shadow-calm-100">
        <div className="text-center">
          <div className="mx-auto w-12 h-12 rounded-2xl bg-wellness-100 flex items-center justify-center text-wellness-700 mb-3 shadow-inner">
            <HeartHandshake className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-calm-900 tracking-tight">Create your safe space</h2>
          <p className="mt-1.5 text-xs sm:text-sm text-calm-500">
            Sign up for autonomous, confidential peer wellness support
          </p>
        </div>

        {errorMessage && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3 text-red-700 text-xs sm:text-sm">
            <AlertCircle className="w-5 h-5 shrink-0 text-red-600" />
            <p>{errorMessage}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-semibold text-calm-700 uppercase tracking-wider mb-1.5">
              Email Address
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-calm-400">
                <Mail className="w-4 h-4" />
              </div>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="student@university.edu"
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-calm-200 focus:outline-none focus:ring-2 focus:ring-wellness-500 focus:border-wellness-500 text-sm text-calm-900 placeholder-calm-400"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-calm-700 uppercase tracking-wider mb-1.5">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-calm-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-calm-200 focus:outline-none focus:ring-2 focus:ring-wellness-500 focus:border-wellness-500 text-sm text-calm-900 placeholder-calm-400"
              />
            </div>

            {/* Password checklist */}
            <div className="mt-2.5 grid grid-cols-2 gap-1.5 text-[11px] text-calm-500 bg-calm-50 p-2.5 rounded-lg border border-calm-100">
              <span className={`flex items-center gap-1 ${pwdChecks.length ? 'text-emerald-700 font-medium' : ''}`}>
                <CheckCircle2 className={`w-3.5 h-3.5 ${pwdChecks.length ? 'text-emerald-600' : 'text-calm-300'}`} />
                8+ characters
              </span>
              <span className={`flex items-center gap-1 ${pwdChecks.uppercase ? 'text-emerald-700 font-medium' : ''}`}>
                <CheckCircle2 className={`w-3.5 h-3.5 ${pwdChecks.uppercase ? 'text-emerald-600' : 'text-calm-300'}`} />
                Uppercase letter
              </span>
              <span className={`flex items-center gap-1 ${pwdChecks.lowercase ? 'text-emerald-700 font-medium' : ''}`}>
                <CheckCircle2 className={`w-3.5 h-3.5 ${pwdChecks.lowercase ? 'text-emerald-600' : 'text-calm-300'}`} />
                Lowercase letter
              </span>
              <span className={`flex items-center gap-1 ${pwdChecks.number ? 'text-emerald-700 font-medium' : ''}`}>
                <CheckCircle2 className={`w-3.5 h-3.5 ${pwdChecks.number ? 'text-emerald-600' : 'text-calm-300'}`} />
                At least 1 number
              </span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-calm-700 uppercase tracking-wider mb-1.5">
              Confirm Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-calm-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-calm-200 focus:outline-none focus:ring-2 focus:ring-wellness-500 focus:border-wellness-500 text-sm text-calm-900 placeholder-calm-400"
              />
            </div>
            {confirmPassword && !doPasswordsMatch && (
              <p className="mt-1 text-[11px] text-red-600">Passwords do not match.</p>
            )}
          </div>

          <div className="flex items-start gap-3 pt-2">
            <input
              type="checkbox"
              id="terms"
              checked={termsAgreed}
              onChange={(e) => setTermsAgreed(e.target.checked)}
              className="mt-1 w-4 h-4 rounded text-wellness-600 focus:ring-wellness-500 border-calm-300"
            />
            <label htmlFor="terms" className="text-xs text-calm-600 leading-relaxed">
              I acknowledge that MindBridge is an AI peer support system, not a therapist or medical provider, and I agree to the{' '}
              <Link to="/consent" className="text-wellness-700 font-medium underline underline-offset-2">
                Privacy Policy & Consent Framework
              </Link>.
            </label>
          </div>

          <button
            type="submit"
            disabled={isLoading || !isPwdValid || !doPasswordsMatch || !termsAgreed}
            className="w-full py-3 px-4 rounded-xl text-white font-medium text-sm bg-wellness-600 hover:bg-wellness-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-wellness-600/20 flex items-center justify-center gap-2 transition-all"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Registering Account...
              </>
            ) : (
              <>
                Register Account
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        <div className="pt-4 border-t border-calm-100 text-center text-xs text-calm-500">
          Already registered?{' '}
          <Link to="/login" className="font-semibold text-wellness-700 hover:text-wellness-800">
            Sign In here
          </Link>
        </div>
      </div>
    </div>
  );
};
