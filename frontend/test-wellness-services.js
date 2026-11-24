import test from 'node:test';
import assert from 'node:assert/strict';

// Test Wellness Resources and Crisis Contacts directly
const TRUSTED_CRISIS_CONTACTS = [
  {
    name: '988 Suicide & Crisis Lifeline',
    contact: 'Call or text 988',
    availability: '24/7, Free & Confidential',
    description: 'Immediate support for anyone experiencing distress or thoughts of self-harm.',
  },
  {
    name: 'Crisis Text Line',
    contact: 'Text HOME to 741741',
    availability: '24/7, Free via SMS',
    description: 'Connect with a crisis counselor via confidential text messaging.',
  },
];

const WELLNESS_RESOURCES = [
  { id: 'box-breathing', title: 'Box Breathing', category: 'stress_management', recommendedFor: ['stress', 'anxiety'] },
  { id: 'grounding-54321', title: '5-4-3-2-1 Grounding', category: 'anxiety_coping', recommendedFor: ['anxiety', 'fear'] },
  { id: 'cognitive-defusion', title: 'Cognitive Reframing', category: 'emotional_regulation', recommendedFor: ['sadness'] },
  { id: 'body-scan', title: 'Progressive Body Scan', category: 'mindfulness', recommendedFor: ['stress'] },
  { id: 'sleep-sanctuary', title: 'Sleep Hygiene', category: 'sleep_hygiene', recommendedFor: ['exhaustion'] },
  { id: 'self-care', title: 'Self-Compassion', category: 'self_care', recommendedFor: ['grief'] },
  { id: 'academic', title: 'Pomodoro Decompression', category: 'academic_stress', recommendedFor: ['overwhelm'] },
  { id: 'social', title: 'Social Reconnection', category: 'social_support', recommendedFor: ['lonely'] },
];

function getRecommendedResources(emotions = [], stressDetected = false, lonelinessDetected = false) {
  const matches = new Set();
  if (stressDetected) {
    WELLNESS_RESOURCES.filter(r => r.category === 'stress_management' || r.category === 'academic_stress').forEach(r => matches.add(r));
  }
  if (lonelinessDetected) {
    WELLNESS_RESOURCES.filter(r => r.category === 'social_support').forEach(r => matches.add(r));
  }
  for (const emotion of emotions) {
    for (const res of WELLNESS_RESOURCES) {
      if (res.recommendedFor.includes(emotion.toLowerCase())) matches.add(res);
    }
  }
  if (matches.size === 0) return [WELLNESS_RESOURCES[0], WELLNESS_RESOURCES[1]];
  return Array.from(matches).slice(0, 3);
}

function processInferenceResult(analysis) {
  const emotionModule = analysis.modules?.emotion;
  const stressModule = analysis.modules?.stress;
  const lonelinessModule = analysis.modules?.loneliness;
  const riskModule = analysis.modules?.risk;

  const detectedEmotions = emotionModule?.confidence_analysis?.selected_labels || [];
  const stressDetected = stressModule?.predicted_class === 'stress';
  const lonelinessDetected = lonelinessModule?.predicted_class === 'lonely';

  const riskCategory = riskModule?.risk_category || 'no_explicit_risk_signal';
  const escalationLevel = riskModule?.escalation_level || 'none';

  const isUrgent = riskCategory === 'urgent_language_signal' || escalationLevel === 'prompt_human_review';
  const isConcerning = riskCategory === 'potential_concern' || escalationLevel === 'human_review';

  let safetyGuidance;
  if (isUrgent) {
    safetyGuidance = {
      isUrgent: true,
      message: 'Immediate support available 24/7.',
      crisisContacts: TRUSTED_CRISIS_CONTACTS,
    };
  } else if (isConcerning) {
    safetyGuidance = {
      isUrgent: false,
      message: 'Supportive resources available.',
      crisisContacts: TRUSTED_CRISIS_CONTACTS.slice(0, 1),
    };
  }

  let supportiveText = '';
  if (isUrgent) {
    supportiveText = "I hear how much pain you are experiencing. Please know you do not have to face this alone.";
  } else if (stressDetected && lonelinessDetected) {
    supportiveText = "It sounds like you are navigating intense pressure and isolation right now.";
  } else if (stressDetected) {
    supportiveText = "I hear how much stress and tension you are carrying.";
  } else if (lonelinessDetected) {
    supportiveText = "Feeling lonely can be an aching experience. Your connection needs matter.";
  } else if (detectedEmotions.includes('joy')) {
    supportiveText = "It is wonderful to hear positive reflections and moments of joy.";
  } else {
    supportiveText = "Thank you for sharing. I am here to listen and reflect with you.";
  }

  const recommendedResources = getRecommendedResources(detectedEmotions, stressDetected, lonelinessDetected);

  return {
    supportiveText,
    detectedEmotions,
    stressDetected,
    lonelinessDetected,
    safetyGuidance,
    recommendedResources,
  };
}

test('Wellness Resources Catalog contains 8 evidence-based categories', () => {
  assert.equal(WELLNESS_RESOURCES.length, 8);
  const categories = new Set(WELLNESS_RESOURCES.map((r) => r.category));
  assert.ok(categories.has('stress_management'));
  assert.ok(categories.has('anxiety_coping'));
  assert.ok(categories.has('mindfulness'));
  assert.ok(categories.has('sleep_hygiene'));
  assert.ok(categories.has('social_support'));
});

test('Crisis contacts include 988 Suicide & Crisis Lifeline', () => {
  const hotline = TRUSTED_CRISIS_CONTACTS.find((c) => c.name.includes('988'));
  assert.ok(hotline, '988 lifeline must be present');
  assert.ok(hotline.contact.includes('988'));
});

test('Resource recommendation handles stress signals', () => {
  const resources = getRecommendedResources(['nervousness'], true, false);
  assert.ok(resources.length > 0);
  assert.ok(
    resources.some((r) => r.category === 'stress_management' || r.category === 'academic_stress')
  );
});

test('Process inference result handles urgent risk category conservatively', () => {
  const mockAnalysis = {
    modules: {
      emotion: { confidence_analysis: { selected_labels: ['sadness'] } },
      stress: { predicted_class: 'stress' },
      loneliness: { predicted_class: 'lonely' },
      risk: { risk_category: 'urgent_language_signal', escalation_level: 'prompt_human_review' },
    },
  };

  const processed = processInferenceResult(mockAnalysis);
  assert.ok(processed.safetyGuidance, 'Safety guidance must be triggered for urgent signals');
  assert.equal(processed.safetyGuidance.isUrgent, true);
  assert.ok(processed.safetyGuidance.crisisContacts.length >= 2);
  assert.ok(processed.supportiveText.includes('pain') || processed.supportiveText.includes('alone'));
});

test('Process inference result handles ordinary calm statement without alarm', () => {
  const mockAnalysis = {
    modules: {
      emotion: { confidence_analysis: { selected_labels: ['joy'] } },
      stress: { predicted_class: 'non_stress' },
      loneliness: { predicted_class: 'non_lonely' },
      risk: { risk_category: 'no_explicit_risk_signal', escalation_level: 'none' },
    },
  };

  const processed = processInferenceResult(mockAnalysis);
  assert.equal(processed.safetyGuidance, undefined);
  assert.ok(processed.supportiveText.includes('joy'));
});
