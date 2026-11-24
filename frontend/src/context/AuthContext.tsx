import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, AuthState } from '../types';
import { api } from '../services/api';

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<{ success: boolean; message: string }>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(() => {
    const saved = localStorage.getItem('mb_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const refreshSession = async (): Promise<boolean> => {
    try {
      const refreshed = await api.refreshToken();
      if (refreshed) {
        setAccessToken(api.getToken());
        // User is authenticated
        return true;
      }
      return false;
    } catch {
      return false;
    }
  };

  useEffect(() => {
    const initAuth = async () => {
      setIsLoading(true);
      await refreshSession();
      setIsLoading(false);
    };
    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      const res = await api.login(email, password);
      setUser(res.user);
      setAccessToken(res.access_token);
      localStorage.setItem('mb_user', JSON.stringify(res.user));
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email: string, password: string) => {
    return api.register(email, password);
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      await api.logout();
    } finally {
      setUser(null);
      setAccessToken(null);
      localStorage.removeItem('mb_user');
      setIsLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        accessToken,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
        refreshSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
