import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { HeartHandshake, Mail, Lock, AlertCircle, ArrowRight, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as any)?.from?.pathname || '/dashboard';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    if (!email.trim() || !password) {
      setErrorMessage('Please enter both email and password.');
      return;
    }

    setIsLoading(true);
    try {
      await login(email.trim(), password);
      navigate(from, { replace: true });
    } catch (err: any) {
      setErrorMessage(err.message || 'Invalid email or password, or account unverified.');
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
          <h2 className="text-2xl font-bold text-calm-900 tracking-tight">Welcome back</h2>
          <p className="mt-1.5 text-xs sm:text-sm text-calm-500">
            Sign in to your private wellness space
          </p>
        </div>

        {errorMessage && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3 text-red-700 text-xs sm:text-sm">
            <AlertCircle className="w-5 h-5 shrink-0 text-red-600" />
            <div className="space-y-1">
              <p>{errorMessage}</p>
              {errorMessage.toLowerCase().includes('unverified') && (
                <Link to="/verify-email" className="block text-xs font-semibold underline text-red-800">
                  Resend verification email
                </Link>
              )}
            </div>
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
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-calm-700 uppercase tracking-wider">
                Password
              </label>
              <Link to="/verify-email" className="text-[11px] text-wellness-700 hover:underline">
                Verification issue?
              </Link>
            </div>
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
          </div>

          <button
            type="submit"
            disabled={isLoading || !email || !password}
            className="w-full py-3 px-4 rounded-xl text-white font-medium text-sm bg-wellness-600 hover:bg-wellness-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-wellness-600/20 flex items-center justify-center gap-2 transition-all"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Signing In...
              </>
            ) : (
              <>
                Sign In
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        <div className="pt-4 border-t border-calm-100 text-center text-xs text-calm-500">
          Don't have an account yet?{' '}
          <Link to="/register" className="font-semibold text-wellness-700 hover:text-wellness-800">
            Sign Up here
          </Link>
        </div>
      </div>
    </div>
  );
};
