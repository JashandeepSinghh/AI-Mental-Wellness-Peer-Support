import type { WellnessResource, CrisisContact } from '../types';

export const TRUSTED_CRISIS_CONTACTS: CrisisContact[] = [
  {
    name: 'Tele-MANAS (Govt. of India)',
    contact: '14416 or 1800-891-4416',
    availability: '24/7, Toll-Free, Multi-lingual (India)',
    description: 'National Tele Mental Health Programme of India (MoHFW & NIMHANS) providing round-the-clock free mental health support and crisis counseling.',
    region: 'India',
  },
  {
    name: 'KIRAN Mental Health Helpline (Govt. of India)',
    contact: '1800-599-0019',
    availability: '24/7, Toll-Free (India)',
    description: 'National 24/7 mental health rehabilitation and psychological first-aid helpline by the Ministry of Social Justice & Empowerment.',
    region: 'India',
  },
  {
    name: 'Vandrevala Foundation Helpline',
    contact: '+91 9999 666 555',
    availability: '24/7, Free & Confidential (India)',
    description: '24/7 tele-counseling and crisis intervention across India in multiple regional languages.',
    region: 'India',
  },
  {
    name: '988 Suicide & Crisis Lifeline',
    contact: 'Call or text 988',
    availability: '24/7, Free & Confidential (US & Canada)',
    description: 'Immediate support for anyone experiencing mental health-related distress, thoughts of suicide, or substance use concerns.',
    region: 'US & Canada',
  },
  {
    name: 'Crisis Text Line',
    contact: 'Text HOME to 741741',
    availability: '24/7, Free via SMS (US, UK, Canada)',
    description: 'Connect with a trained crisis counselor via confidential text messaging for immediate support in any crisis.',
    region: 'US & Canada',
  },
  {
    name: 'The Trevor Project (LGBTQ Youth)',
    contact: '1-866-488-7386 or Text START to 678-678',
    availability: '24/7, Free & Confidential (LGBTQ Youth)',
    description: 'Specialized crisis intervention and suicide prevention services for lesbian, gay, bisexual, transgender, queer & questioning young people.',
    region: 'US & Canada',
  },
  {
    name: 'Befrienders Worldwide',
    contact: 'https://www.befrienders.org',
    availability: 'Global directory of confidential emotional support',
    description: 'Find verified local emotional support hotlines and mental health peer networks across 32+ countries.',
    region: 'International',
  },
  {
    name: 'Find A Helpline',
    contact: 'https://findahelpline.com',
    availability: 'Global Directory (130+ countries)',
    description: 'Free, confidential emotional support and local crisis lines worldwide sorted by country.',
    region: 'International',
  },
];

export const WELLNESS_RESOURCES: WellnessResource[] = [
  {
    id: 'box-breathing',
    title: 'Box Breathing Technique for Acute Stress',
    category: 'stress_management',
    summary: 'A 4-part rhythmic breathing exercise used to down-regulate the sympathetic nervous system and rapidly reduce stress hormones.',
    guidanceSteps: [
      'Inhale slowly through your nose for a count of 4 seconds.',
      'Hold your breath gently for 4 seconds without straining.',
      'Exhale smoothly through your mouth for 4 seconds.',
      'Pause and hold your lungs empty for 4 seconds.',
      'Repeat the 4-4-4-4 cycle for 3 to 5 minutes until you feel grounded.',
    ],
    recommendedFor: ['stress', 'nervousness', 'anxiety', 'panic', 'overwhelm'],
    readTimeMinutes: 3,
  },
  {
    id: 'grounding-54321',
    title: '5-4-3-2-1 Sensory Grounding Method',
    category: 'anxiety_coping',
    summary: 'Reconnect with your physical environment and interrupt anxious rumination by engaging your five senses.',
    guidanceSteps: [
      'Acknowledge 5 things you can see around you right now (colors, shapes, objects).',
      'Acknowledge 4 things you can physically feel (feet on floor, fabric of your shirt, breeze).',
      'Acknowledge 3 distinct sounds you can hear (clock ticking, traffic, your own breath).',
      'Acknowledge 2 things you can smell or scents you enjoy.',
      'Acknowledge 1 thing you can taste or take a mindful sip of water.',
    ],
    recommendedFor: ['anxiety', 'fear', 'nervousness', 'rumination', 'confusion'],
    readTimeMinutes: 4,
  },
  {
    id: 'cognitive-defusion',
    title: 'Cognitive Reframing & Thought Labeling',
    category: 'emotional_regulation',
    summary: 'Notice unhelpful thought loops without judgment by treating thoughts as transient mental events rather than absolute facts.',
    guidanceSteps: [
      'Notice the thought causing distress (e.g., "I will fail").',
      'Prefix the thought with: "I am having the thought that I will fail."',
      'Deepen awareness: "I notice that I am experiencing the thought that I will fail."',
      'Observe what emotion arises in your body without trying to push it away.',
      'Ask: "Does holding onto this thought serve me right now, or can I let it pass like a cloud?"',
    ],
    recommendedFor: ['disappointment', 'disapproval', 'remorse', 'sadness', 'annoyance'],
    readTimeMinutes: 5,
  },
  {
    id: 'body-scan-mindfulness',
    title: 'Progressive Body Scan & Muscle Relaxation',
    category: 'mindfulness',
    summary: 'Systematically release stored tension across muscle groups to restore mental clarity and physical equilibrium.',
    guidanceSteps: [
      'Find a comfortable seated position with your spine supported and feet uncrossed.',
      'Close your eyes or soften your gaze; take 3 deep, steady breaths.',
      'Bring gentle attention to your forehead and jaw, consciously letting them soften.',
      'Move awareness down through shoulders, arms, chest, stomach, hips, and legs.',
      'Release any areas of tightness with each exhalation.',
    ],
    recommendedFor: ['stress', 'anger', 'nervousness', 'exhaustion'],
    readTimeMinutes: 6,
  },
  {
    id: 'sleep-sanctuary',
    title: 'Evidence-Based Sleep Hygiene Protocol',
    category: 'sleep_hygiene',
    summary: 'Restore natural circadian rhythms to improve sleep architecture, emotional resilience, and cognitive endurance.',
    guidanceSteps: [
      'Establish a consistent wake-up time 7 days a week to anchor your circadian rhythm.',
      'Avoid bright screens and blue light 45–60 minutes before bedtime.',
      'Keep your sleeping environment cool (around 65°F / 18°C), dark, and quiet.',
      'Reserve your bed solely for sleep and rest so your brain builds strong associative sleep cues.',
      'If you cannot sleep after 20 minutes, get up and engage in a low-stimulus activity in dim light.',
    ],
    recommendedFor: ['exhaustion', 'stress', 'sadness', 'overwhelm'],
    readTimeMinutes: 5,
  },
  {
    id: 'micro-self-care',
    title: 'Micro-Moments of Self-Compassion',
    category: 'self_care',
    summary: 'Simple, accessible practices for treating yourself with kindness during challenging or emotionally exhausting moments.',
    guidanceSteps: [
      'Place a comforting hand over your chest or upper arm to activate mammalian soothing pathways.',
      'Silently say to yourself: "This is a moment of difficulty. Difficulty is a normal part of life."',
      'Ask yourself: "What kind, supportive words would I offer to a close friend in this exact situation?"',
      'Offer those exact words of warmth, validation, and encouragement to yourself.',
    ],
    recommendedFor: ['grief', 'sadness', 'embarrassment', 'disappointment', 'remorse'],
    readTimeMinutes: 3,
  },
  {
    id: 'pomodoro-decompression',
    title: 'Managing Academic Overwhelm & Procrastination',
    category: 'academic_stress',
    summary: 'Break overwhelming workloads into manageable micro-commitments to bypass anxiety-driven task avoidance.',
    guidanceSteps: [
      'Select one single, specific task (not "study for exam", but "review 5 pages of chapter 2").',
      'Commit to working for just 15 to 20 minutes without interruption or switching tabs.',
      'Notice that the hardest part is task initiation; after 5 minutes, resistance naturally lowers.',
      'Take a mandatory 5-minute break away from screens after each focused interval.',
      'Celebrate completing the interval rather than demanding perfection.',
    ],
    recommendedFor: ['stress', 'annoyance', 'confusion', 'disappointment'],
    readTimeMinutes: 4,
  },
  {
    id: 'loneliness-connection',
    title: 'Rebuilding Social Connectedness in Steps',
    category: 'social_support',
    summary: 'Gentle, low-stakes pathways to alleviate isolation and foster genuine belonging.',
    guidanceSteps: [
      'Acknowledge that feeling lonely is a universal human signal that your social needs need nurturing.',
      'Reach out to one acquaintance or family member with a short, low-pressure message (e.g., "Thinking of you, hope your week is going well").',
      'Engage in shared ambient spaces (a campus library, café, community park) to experience "ambient belonging".',
      'Look for interest-based university groups or peer-support forums where shared hobbies bridge initial barriers.',
    ],
    recommendedFor: ['lonely', 'sadness', 'grief', 'curiosity'],
    readTimeMinutes: 5,
  },
];

export function getRecommendedResources(
  emotions: string[] = [],
  stressDetected = false,
  lonelinessDetected = false
): WellnessResource[] {
  const matches = new Set<WellnessResource>();

  if (stressDetected) {
    WELLNESS_RESOURCES.filter(r => r.category === 'stress_management' || r.category === 'academic_stress')
      .forEach(r => matches.add(r));
  }

  if (lonelinessDetected) {
    WELLNESS_RESOURCES.filter(r => r.category === 'social_support')
      .forEach(r => matches.add(r));
  }

  for (const emotion of emotions) {
    for (const res of WELLNESS_RESOURCES) {
      if (res.recommendedFor.includes(emotion.toLowerCase())) {
        matches.add(res);
      }
    }
  }

  if (matches.size === 0) {
    // Return sensible defaults
    return [WELLNESS_RESOURCES[0], WELLNESS_RESOURCES[1], WELLNESS_RESOURCES[5]];
  }

  return Array.from(matches).slice(0, 3);
}
