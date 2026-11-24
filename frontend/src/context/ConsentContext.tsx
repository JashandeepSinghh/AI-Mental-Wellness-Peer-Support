import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { ConsentRecord } from '../types';
import { api } from '../services/api';
import { useAuth } from './AuthContext';

interface ConsentContextType {
  conversationStorage: ConsentRecord | null;
  analytics: ConsentRecord | null;
  isLoading: boolean;
  error: string | null;
  updateConsent: (purpose: 'conversation_storage' | 'analytics', granted: boolean) => Promise<void>;
  reloadConsents: () => Promise<void>;
}

const ConsentContext = createContext<ConsentContextType | undefined>(undefined);

export const ConsentProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  const [conversationStorage, setConversationStorage] = useState<ConsentRecord | null>(null);
  const [analytics, setAnalytics] = useState<ConsentRecord | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const reloadConsents = useCallback(async () => {
    if (!isAuthenticated) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.getConsents();
      if (res.success && res.consents) {
        setConversationStorage(res.consents.conversation_storage);
        setAnalytics(res.consents.analytics);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load consent settings');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (isAuthenticated) {
      reloadConsents();
    } else {
      setConversationStorage(null);
      setAnalytics(null);
    }
  }, [isAuthenticated, reloadConsents]);

  const updateConsent = async (purpose: 'conversation_storage' | 'analytics', granted: boolean) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.recordConsent(purpose, granted, '1.0');
      if (res.success && res.consent) {
        if (purpose === 'conversation_storage') {
          setConversationStorage(res.consent);
        } else {
          setAnalytics(res.consent);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to record consent decision');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ConsentContext.Provider
      value={{
        conversationStorage,
        analytics,
        isLoading,
        error,
        updateConsent,
        reloadConsents,
      }}
    >
      {children}
    </ConsentContext.Provider>
  );
};

export const useConsent = () => {
  const context = useContext(ConsentContext);
  if (!context) {
    throw new Error('useConsent must be used within a ConsentProvider');
  }
  return context;
};
