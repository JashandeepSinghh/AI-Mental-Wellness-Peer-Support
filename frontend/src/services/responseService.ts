import type { InferenceAnalysis, SafetyGuidance, WellnessResource } from '../types';
import { TRUSTED_CRISIS_CONTACTS, getRecommendedResources } from './wellnessResources';

export interface ProcessedResponse {
  supportiveText: string;
  detectedEmotions: string[];
  sentiment: string;
  stressDetected: boolean;
  lonelinessDetected: boolean;
  safetyGuidance?: SafetyGuidance;
  recommendedResources: WellnessResource[];
}

export function processInferenceResult(analysis: InferenceAnalysis): ProcessedResponse {
  const emotionModule = analysis.modules?.emotion;
  const sentimentModule = analysis.modules?.sentiment;
  const stressModule = analysis.modules?.stress;
  const lonelinessModule = analysis.modules?.loneliness;
  const riskModule = analysis.modules?.risk;

  // Extract selected or top emotions
  let detectedEmotions: string[] = [];
  if (emotionModule?.confidence_analysis?.selected_labels && emotionModule.confidence_analysis.selected_labels.length > 0) {
    detectedEmotions = emotionModule.confidence_analysis.selected_labels;
  } else if (emotionModule?.predictions) {
    detectedEmotions = Object.entries(emotionModule.predictions)
      .filter(([name, prob]) => prob > 0.15 && name !== 'neutral')
      .sort((a, b) => b[1] - a[1])
      .map(([name]) => name)
      .slice(0, 3);
  }

  const sentiment = sentimentModule?.predicted_class || 'neutral';
  const stressDetected = stressModule?.predicted_class === 'stress';
  const lonelinessDetected = lonelinessModule?.predicted_class === 'lonely';

  const riskCategory = riskModule?.risk_category || 'no_explicit_risk_signal';
  const escalationLevel = riskModule?.escalation_level || 'none';

  const isUrgent =
    riskCategory === 'urgent_language_signal' ||
    escalationLevel === 'prompt_human_review';

  const isConcerning =
    riskCategory === 'potential_concern' ||
    escalationLevel === 'human_review' ||
    escalationLevel === 'supportive_follow_up';

  let safetyGuidance: SafetyGuidance | undefined;

  if (isUrgent) {
    safetyGuidance = {
      isUrgent: true,
      isConcerning: true,
      message:
        'Your well-being is deeply important. Because you may be experiencing intense pain or distress right now, please know that you do not have to carry this alone. Free, confidential support from compassionate professionals is available 24/7.',
      crisisContacts: TRUSTED_CRISIS_CONTACTS,
    };
  } else if (isConcerning) {
    safetyGuidance = {
      isUrgent: false,
      isConcerning: true,
      message:
        'It sounds like you are carrying a heavy burden right now. While I am an AI peer support companion and cannot provide therapy or clinical diagnosis, supportive resources and professional care are always here for you.',
      crisisContacts: TRUSTED_CRISIS_CONTACTS.slice(0, 2),
    };
  }

  // Construct supportive empathetic response
  let supportiveText = '';

  if (isUrgent) {
    supportiveText =
      "I hear how much pain and weight you are experiencing in this moment. Please remember that what you are feeling is valid, but you don't have to face this by yourself. Connecting with someone who is trained to support you through moments like this can help make things clearer and safer.";
  } else if (stressDetected && lonelinessDetected) {
    supportiveText =
      "It sounds like you're navigating both intense pressure and feeling isolated right now. Carrying a lot of stress while feeling like there's no one around to share it with is exhausting. I want to acknowledge how much courage it takes to put that into words. Let's take things one moment at a time.";
  } else if (stressDetected) {
    supportiveText =
      "I hear how much tension and stress you're experiencing right now. When demands feel high, our minds and bodies often tense up in response. It is completely understandable to feel overwhelmed. Would it help to take a slow breath together or explore a small coping step?";
  } else if (lonelinessDetected) {
    supportiveText =
      "Feeling disconnected or lonely can be such a quiet, aching experience. Thank you for sharing that with me. Please remember that feeling lonely doesn't mean you are unworthy of connection—it's simply a sign that your social and emotional needs deserve care.";
  } else if (detectedEmotions.includes('sadness') || detectedEmotions.includes('grief')) {
    supportiveText =
      "I notice feelings of sadness in what you shared. It is completely okay to feel down, and you don't need to force yourself to feel positive right away. Giving yourself permission to simply feel what you are feeling is often the first gentle step toward healing.";
  } else if (detectedEmotions.includes('fear') || detectedEmotions.includes('nervousness')) {
    supportiveText =
      "It sounds like anxiety or uncertainty is weighing on your mind. When apprehension takes over, grounding yourself in the present moment can bring some stability back. Take a deep, gentle breath with me.";
  } else if (detectedEmotions.includes('joy') || detectedEmotions.includes('gratitude') || detectedEmotions.includes('optimism')) {
    supportiveText =
      "It's really heartening to hear positive reflections and moments of gratitude in your message. Savoring these moments of warmth builds lasting emotional resilience.";
  } else {
    supportiveText =
      "Thank you for sharing your thoughts with me. I'm here to listen, support, and help you reflect on whatever you are experiencing right now.";
  }

  const recommendedResources = getRecommendedResources(
    detectedEmotions,
    stressDetected,
    lonelinessDetected
  );

  return {
    supportiveText,
    detectedEmotions,
    sentiment,
    stressDetected,
    lonelinessDetected,
    safetyGuidance,
    recommendedResources,
  };
}
