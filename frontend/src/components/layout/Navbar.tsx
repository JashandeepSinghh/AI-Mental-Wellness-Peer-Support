import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { HeartHandshake, ShieldCheck, Activity, MessageSquareHeart, BookOpen, LogOut, User } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export const Navbar: React.FC = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-calm-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo & Brand */}
          <Link to={isAuthenticated ? "/dashboard" : "/"} className="flex items-center gap-2.5 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-wellness-600 to-wellness-400 flex items-center justify-center text-white shadow-md shadow-wellness-500/20 group-hover:scale-105 transition-transform">
              <HeartHandshake className="w-6 h-6" />
            </div>
            <div>
              <span className="font-semibold text-lg text-calm-900 tracking-tight flex items-center gap-1.5">
                MindBridge
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 bg-wellness-100 text-wellness-800 rounded-full">AI Peer</span>
              </span>
              <p className="text-xs text-calm-500 leading-none">Support & Wellness</p>
            </div>
          </Link>

          {/* Navigation Links */}
          {isAuthenticated ? (
            <nav className="hidden md:flex items-center gap-1">
              <Link
                to="/dashboard"
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  isActive('/dashboard')
                    ? 'bg-wellness-50 text-wellness-700'
                    : 'text-calm-600 hover:text-calm-900 hover:bg-calm-100'
                }`}
              >
                <Activity className="w-4 h-4" />
                Dashboard
              </Link>
              <Link
                to="/chat"
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  isActive('/chat')
                    ? 'bg-wellness-50 text-wellness-700'
                    : 'text-calm-600 hover:text-calm-900 hover:bg-calm-100'
                }`}
              >
                <MessageSquareHeart className="w-4 h-4" />
                AI Wellness Chat
              </Link>
              <Link
                to="/resources"
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  isActive('/resources')
                    ? 'bg-wellness-50 text-wellness-700'
                    : 'text-calm-600 hover:text-calm-900 hover:bg-calm-100'
                }`}
              >
                <BookOpen className="w-4 h-4" />
                Resources
              </Link>
              <Link
                to="/analytics"
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  isActive('/analytics')
                    ? 'bg-wellness-50 text-wellness-700'
                    : 'text-calm-600 hover:text-calm-900 hover:bg-calm-100'
                }`}
              >
                <Activity className="w-4 h-4" />
                Analytics
              </Link>
              <Link
                to="/consent"
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  isActive('/consent')
                    ? 'bg-wellness-50 text-wellness-700'
                    : 'text-calm-600 hover:text-calm-900 hover:bg-calm-100'
                }`}
              >
                <ShieldCheck className="w-4 h-4" />
                Privacy & Consent
              </Link>
            </nav>
          ) : (
            <div className="hidden md:flex items-center gap-4 text-sm font-medium text-calm-600">
              <Link to="/chat" className={`hover:text-calm-900 transition-colors ${isActive('/chat') || isActive('/') ? 'text-wellness-700 font-semibold' : ''}`}>Wellness Chat</Link>
              <Link to="/resources" className={`hover:text-calm-900 transition-colors ${isActive('/resources') ? 'text-wellness-700 font-semibold' : ''}`}>Resources</Link>
              <Link to="/privacy-info" className={`hover:text-calm-900 transition-colors ${isActive('/privacy-info') ? 'text-wellness-700 font-semibold' : ''}`}>Safety & Ethics</Link>
            </div>
          )}

          {/* User Controls */}
          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <div className="flex items-center gap-2">
                <Link
                  to="/profile"
                  className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-calm-200 bg-calm-50 text-xs text-calm-700 hover:bg-calm-100 transition-colors"
                >
                  <User className="w-3.5 h-3.5 text-wellness-600" />
                  <span className="font-medium capitalize">{user?.role || 'Student'}</span>
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-1.5 text-xs font-medium text-calm-600 hover:text-red-600 px-3 py-2 rounded-lg hover:bg-red-50 transition-colors"
                  title="Sign out safely"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="hidden sm:inline">Sign Out</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Link
                  to="/login"
                  className="text-sm font-medium text-calm-700 hover:text-wellness-700 px-3 py-2 rounded-lg hover:bg-calm-100 transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  className="text-sm font-medium text-white bg-wellness-600 hover:bg-wellness-700 px-4 py-2 rounded-lg shadow-sm shadow-wellness-600/30 transition-all hover:shadow-md"
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
