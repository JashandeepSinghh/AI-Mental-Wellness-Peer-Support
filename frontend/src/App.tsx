import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ConsentProvider } from './context/ConsentContext';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { ProtectedRoute } from './components/common/ProtectedRoute';

import { LandingPage } from './features/landing/LandingPage';
import { RegisterPage } from './features/auth/RegisterPage';
import { LoginPage } from './features/auth/LoginPage';
import { VerifyEmailPage } from './features/auth/VerifyEmailPage';
import { DashboardPage } from './features/dashboard/DashboardPage';
import { ChatPage } from './features/chat/ChatPage';
import { ResourcesPage } from './features/resources/ResourcesPage';
import { ConsentPage } from './features/consent/ConsentPage';
import { AnalyticsPage } from './features/analytics/AnalyticsPage';
import { ProfilePage } from './features/profile/ProfilePage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ConsentProvider>
          <div className="flex flex-col min-h-screen">
            <Navbar />
            <main className="flex-1">
              <Routes>
                {/* Public Immediate-Access Routes */}
                <Route path="/" element={<ChatPage />} />
                <Route path="/chat" element={<ChatPage />} />
                <Route path="/landing" element={<LandingPage />} />
                <Route path="/about" element={<LandingPage />} />
                <Route path="/privacy-info" element={<LandingPage />} />
                <Route path="/resources" element={<ResourcesPage />} />
                <Route path="/register" element={<RegisterPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/verify-email" element={<VerifyEmailPage />} />
                <Route path="/verify-email-instruction" element={<VerifyEmailPage />} />

                {/* Authenticated Protected Routes */}
                <Route
                  path="/dashboard"
                  element={
                    <ProtectedRoute>
                      <DashboardPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/consent"
                  element={
                    <ProtectedRoute>
                      <ConsentPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/analytics"
                  element={
                    <ProtectedRoute>
                      <AnalyticsPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/profile"
                  element={
                    <ProtectedRoute>
                      <ProfilePage />
                    </ProtectedRoute>
                  }
                />

                {/* Catch-all redirect */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
            <Footer />
          </div>
        </ConsentProvider>
      </AuthProvider>
    </BrowserRouter>
  );
};

export default App;
