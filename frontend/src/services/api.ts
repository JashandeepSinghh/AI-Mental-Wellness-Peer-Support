// API Client service for MindBridge Mental Wellness & Peer Support Platform

const API_BASE = '/api/v1';

class ApiService {
  private accessToken: string | null = null;

  setToken(token: string | null) {
    this.accessToken = token;
  }

  getToken(): string | null {
    return this.accessToken;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const headers = new Headers(options.headers || {});

    if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
    }

    if (this.accessToken) {
      headers.set('Authorization', `Bearer ${this.accessToken}`);
    }

    const config: RequestInit = {
      ...options,
      headers,
      credentials: 'include', // Ensures HttpOnly cookies (refresh_token) are transmitted
    };

    const response = await fetch(url, config);

    if (response.status === 401 && endpoint !== '/auth/login' && endpoint !== '/auth/refresh') {
      // Attempt refresh
      const refreshed = await this.refreshToken();
      if (refreshed) {
        headers.set('Authorization', `Bearer ${this.accessToken}`);
        return (await fetch(url, { ...config, headers })).json();
      }
    }

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const errorMsg = data?.detail || data?.message || data?.error?.message || `HTTP ${response.status}`;
      throw new Error(errorMsg);
    }

    return data as T;
  }

  // Auth Endpoints
  async register(email: string, password: string):Promise<{ success: boolean; message: string; status: string }> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async login(email: string, password: string): Promise<{
    success: boolean;
    access_token: string;
    token_type: string;
    expires_in: number;
    user: { id: string; role: any };
  }> {
    const res = await this.request<any>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    if (res.access_token) {
      this.setToken(res.access_token);
    }
    return res;
  }

  async refreshToken(): Promise<boolean> {
    try {
      const res = await this.request<any>('/auth/refresh', {
        method: 'POST',
      });
      if (res.access_token) {
        this.setToken(res.access_token);
        return true;
      }
      return false;
    } catch {
      this.setToken(null);
      return false;
    }
  }

  async logout(): Promise<void> {
    try {
      await this.request('/auth/logout', { method: 'POST' });
    } finally {
      this.setToken(null);
    }
  }

  async verifyEmail(token: string): Promise<{ success: boolean; message: string }> {
    return this.request('/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ token }),
    });
  }

  async resendVerification(email: string): Promise<{ success: boolean; message: string }> {
    return this.request('/auth/resend-verification', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  }

  // Consent Endpoints
  async getConsents(): Promise<{
    success: boolean;
    consents: {
      conversation_storage: any;
      analytics: any;
    };
  }> {
    return this.request('/consent', { method: 'GET' });
  }

  async recordConsent(
    purpose: 'conversation_storage' | 'analytics',
    granted: boolean,
    policy_version = '1.0'
  ): Promise<{ success: boolean; message: string; consent: any }> {
    return this.request('/consent', {
      method: 'POST',
      body: JSON.stringify({
        purpose,
        granted,
        policy_version,
      }),
    });
  }

  // Chat Endpoint
  async sendChatMessage(
    message: string,
    conversation_id?: string
  ): Promise<{
    success: boolean;
    status: string;
    message: string;
    conversation_id: string | null;
    message_length: number;
    analysis: any;
  }> {
    return this.request('/chat/messages', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversation_id || undefined,
        metadata: {
          platform: 'web',
          app_version: '1.0.0',
        },
      }),
    });
  }

  // Health Endpoint
  async checkHealth(): Promise<{ status: string; service: string; version: string }> {
    return this.request('/health', { method: 'GET' });
  }
}

export const api = new ApiService();
