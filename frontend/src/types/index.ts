// Type definitions for MindBridge Mental Wellness & Peer Support Platform

export type Role = 'student' | 'admin' | 'counsellor';

export interface User {
  id: string;
  role: Role;
  email?: string;
}

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface ConsentRecord {
  id: string;
  granted: boolean;
  policy_version: string;
  source: string;
  recorded_at: string;
}

export interface ConsentsResponse {
  success: boolean;
  consents: {
    conversation_storage: ConsentRecord | null;
    analytics: ConsentRecord | null;
  };
}

export interface EmotionConfidence {
  output_type: string;
  predicted_class?: string;
  confidence: number;
  uncertainty_band?: string;
  prediction_margin?: number;
  selected_labels?: string[];
  label_results?: Array<{
    label_name: string;
    probability: number;
    threshold: number;
    predicted_positive: boolean;
  }>;
}

export interface RiskAnalysis {
  module_name: string;
  risk_category: 'no_explicit_risk_signal' | 'potential_concern' | 'urgent_language_signal' | 'uncertain_or_ambiguous';
  escalation_level: 'none' | 'supportive_follow_up' | 'human_review' | 'prompt_human_review';
  detected_indicators: Record<string, any>;
  context_flags: {
    negation?: { negation_detected: boolean };
    temporal?: { historical_reference_detected: boolean; current_time_reference_detected: boolean };
    hypothetical?: { hypothetical_detected: boolean };
    third_person?: { third_person_reference_detected: boolean };
    quotation?: { quotation_detected: boolean };
  };
  explanation?: string;
  requires_human_review?: boolean;
}

export interface InferenceAnalysis {
  status: string;
  input_length: number;
  modules: {
    emotion?: {
      status: string;
      probabilities?: number[];
      predictions?: Record<string, number>;
      selected_labels?: string[];
      confidence_analysis?: EmotionConfidence;
    };
    sentiment?: {
      status: string;
      predicted_class?: 'positive' | 'neutral' | 'negative';
      probabilities?: Record<string, number>;
      confidence_analysis?: EmotionConfidence;
    };
    stress?: {
      status: string;
      predicted_class?: 'stress' | 'non_stress';
      confidence_analysis?: EmotionConfidence;
    };
    loneliness?: {
      status: string;
      predicted_class?: 'lonely' | 'non_lonely';
      confidence_analysis?: EmotionConfidence;
    };
    risk?: RiskAnalysis;
    rumination?: {
      status: string;
      rumination_detected?: boolean;
      score?: number;
    };
    vad?: {
      status: string;
      valence?: number;
      arousal?: number;
      dominance?: number;
    };
  };
  confidence_report?: {
    module_count?: number;
    evaluated_module_count?: number;
    modules?: Record<string, any>;
  };
  disclaimers?: string[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant' | 'system';
  text: string;
  timestamp: string;
  analysis?: InferenceAnalysis;
  supportiveResponse?: string;
  suggestedResources?: WellnessResource[];
  safetyGuidance?: SafetyGuidance;
}

export interface SafetyGuidance {
  isUrgent: boolean;
  isConcerning: boolean;
  message: string;
  crisisContacts: CrisisContact[];
}

export interface CrisisContact {
  name: string;
  contact: string;
  availability: string;
  description: string;
  region?: string;
}

export interface WellnessResource {
  id: string;
  title: string;
  category: ResourceCategory;
  summary: string;
  guidanceSteps: string[];
  recommendedFor: string[];
  readTimeMinutes: number;
}

export type ResourceCategory =
  | 'stress_management'
  | 'anxiety_coping'
  | 'emotional_regulation'
  | 'mindfulness'
  | 'sleep_hygiene'
  | 'self_care'
  | 'academic_stress'
  | 'social_support';
