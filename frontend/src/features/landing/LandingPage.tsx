import React from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  Shield,
  Heart,
  Brain,
  Lock,
  ArrowRight,
  Compass,
  MessageCircle,
  Activity,
  PhoneCall,
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div className="space-y-20 pb-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 md:pt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-wellness-100 text-wellness-800 text-xs sm:text-sm font-medium mb-6 animate-fade-in border border-wellness-200">
            <Sparkles className="w-4 h-4 text-wellness-600" />
            <span>Autonomous AI Wellness & Empathetic Peer Support</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-calm-950 tracking-tight max-w-4xl mx-auto leading-tight">
            A Safe, Calming Space to <span className="text-wellness-600 underline decoration-wellness-300 decoration-wavy decoration-2">Reflect</span> and Feel Heard.
          </h1>

          <p className="mt-6 text-base sm:text-lg text-calm-600 max-w-2xl mx-auto leading-relaxed">
            MindBridge couples multi-dimensional deep learning models with compassionate conversational guidance to help students navigate emotional overwhelm, anxiety, stress, and loneliness.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/register"
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl bg-wellness-600 hover:bg-wellness-700 text-white font-medium text-base shadow-lg shadow-wellness-600/25 transition-all hover:-translate-y-0.5"
            >
              Get Started for Free
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              to="/chat"
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl bg-white border border-calm-200 hover:bg-calm-50 text-calm-800 font-medium text-base shadow-sm transition-all"
            >
              <MessageCircle className="w-5 h-5 text-wellness-600" />
              Try Wellness Chat
            </Link>
          </div>

          {/* Privacy Trust Badges */}
          <div className="mt-12 flex flex-wrap items-center justify-center gap-6 text-xs sm:text-sm text-calm-500 font-medium">
            <div className="flex items-center gap-2">
              <Lock className="w-4 h-4 text-wellness-600" />
              <span>Zero-Message Server Storage</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-wellness-600" />
              <span>Tamper-Evident Consent Logging</span>
            </div>
            <div className="flex items-center gap-2">
              <Heart className="w-4 h-4 text-wellness-600" />
              <span>100% Free & Student-Centric</span>
            </div>
          </div>
        </div>

        {/* Subtle decorative background blur */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-wellness-200/40 rounded-full blur-3xl -z-10 pointer-events-none" />
      </section>

      {/* Core Capabilities */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-xs uppercase font-bold tracking-widest text-wellness-700 mb-2">Multimodal NLP Architecture</h2>
          <p className="text-3xl font-bold text-calm-950">Engineered Specifically for Emotional Well-Being</p>
          <p className="mt-3 text-calm-600 text-sm sm:text-base">
            Unlike generic chatbots, MindBridge uses specialized neural networks fine-tuned on psychological and emotional corpora.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-wellness-50 text-wellness-600 flex items-center justify-center mb-5">
              <Brain className="w-6 h-6" />
            </div>
            <h3 className="font-semibold text-calm-900 text-lg mb-2">Multi-Label Emotion Inference</h3>
            <p className="text-sm text-calm-600 leading-relaxed">
              Detects nuances across 28 distinct emotional states using our validation-calibrated GoEmotions DistilBERT transformer.
            </p>
            <div className="mt-4 flex flex-wrap gap-1.5">
              <span className="text-xs px-2 py-0.5 bg-calm-100 text-calm-700 rounded-md">Joy</span>
              <span className="text-xs px-2 py-0.5 bg-calm-100 text-calm-700 rounded-md">Anxiety</span>
              <span className="text-xs px-2 py-0.5 bg-calm-100 text-calm-700 rounded-md">Grief</span>
              <span className="text-xs px-2 py-0.5 bg-calm-100 text-calm-700 rounded-md">+25 more</span>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-wellness-50 text-wellness-600 flex items-center justify-center mb-5">
              <Activity className="w-6 h-6" />
            </div>
            <h3 className="font-semibold text-calm-900 text-lg mb-2">Stress & Loneliness Triage</h3>
            <p className="text-sm text-calm-600 leading-relaxed">
              Specialized models analyze acute academic distress and social isolation signals to provide targeted coping interventions before burnout occurs.
            </p>
            <div className="mt-4 flex flex-wrap gap-1.5">
              <span className="text-xs px-2 py-0.5 bg-wellness-50 text-wellness-700 rounded-md">Academic Stress</span>
              <span className="text-xs px-2 py-0.5 bg-wellness-50 text-wellness-700 rounded-md">Social Isolation</span>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-wellness-50 text-wellness-600 flex items-center justify-center mb-5">
              <Shield className="w-6 h-6" />
            </div>
            <h3 className="font-semibold text-calm-900 text-lg mb-2">Multi-Tier Safety & Escalation</h3>
            <p className="text-sm text-calm-600 leading-relaxed">
              Synthesizes sentence-level context, negation, and temporal signals to identify urgent language and provide immediate, confidential crisis resources.
            </p>
            <div className="mt-4 flex flex-wrap gap-1.5">
              <span className="text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-md">24/7 Lifeline Ready</span>
              <span className="text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-md">Context-Aware</span>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-wellness-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-xs uppercase font-bold tracking-widest text-wellness-300 mb-2">Simple & Respectful</h2>
            <p className="text-3xl font-bold">How MindBridge Works</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="bg-wellness-800/60 p-6 rounded-xl border border-wellness-700/50">
              <span className="text-2xl font-black text-wellness-300">01</span>
              <h3 className="font-semibold text-lg mt-2 mb-2">Express Yourself</h3>
              <p className="text-xs sm:text-sm text-wellness-100 leading-relaxed">
                Type what you are experiencing into the confidential wellness chat—freely and without judgment.
              </p>
            </div>

            <div className="bg-wellness-800/60 p-6 rounded-xl border border-wellness-700/50">
              <span className="text-2xl font-black text-wellness-300">02</span>
              <h3 className="font-semibold text-lg mt-2 mb-2">In-Memory AI Analysis</h3>
              <p className="text-xs sm:text-sm text-wellness-100 leading-relaxed">
                The unified pipeline extracts emotional states and evaluates stress levels instantly without writing raw text to database logs.
              </p>
            </div>

            <div className="bg-wellness-800/60 p-6 rounded-xl border border-wellness-700/50">
              <span className="text-2xl font-black text-wellness-300">03</span>
              <h3 className="font-semibold text-lg mt-2 mb-2">Empathetic Reflection</h3>
              <p className="text-xs sm:text-sm text-wellness-100 leading-relaxed">
                Receive thoughtful, non-judgmental validation and personalized grounding steps tailored to your current state.
              </p>
            </div>

            <div className="bg-wellness-800/60 p-6 rounded-xl border border-wellness-700/50">
              <span className="text-2xl font-black text-wellness-300">04</span>
              <h3 className="font-semibold text-lg mt-2 mb-2">Support & Coping</h3>
              <p className="text-xs sm:text-sm text-wellness-100 leading-relaxed">
                Access curated somatic exercises, sleep protocols, and verified crisis helplines whenever you need them.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Safety & Crisis Reassurance */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-gradient-to-br from-wellness-50 to-sage-50 border border-wellness-200 rounded-3xl p-8 sm:p-10 shadow-sm flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-wellness-200/60 text-wellness-900 rounded-md text-xs font-semibold">
              <PhoneCall className="w-4 h-4 text-wellness-700" />
              <span>Immediate Support Network</span>
            </div>
            <h3 className="text-2xl font-bold text-calm-900">Are you or someone you know going through a crisis?</h3>
            <p className="text-sm text-calm-600 max-w-xl leading-relaxed">
              You are never alone. Professional counselors are available around the clock to offer free, confidential guidance. No appointment is needed.
            </p>
            <div className="flex flex-wrap gap-3 pt-2">
              <a
                href="tel:14416"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-medium text-sm transition-colors shadow-sm"
                title="Tele-MANAS (India National Mental Health Helpline)"
              >
                <PhoneCall className="w-4 h-4" />
                India: 14416 (Tele-MANAS)
              </a>
              <a
                href="tel:988"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-wellness-700 hover:bg-wellness-800 text-white font-medium text-sm transition-colors shadow-sm"
                title="988 Suicide & Crisis Lifeline (US & Canada)"
              >
                <PhoneCall className="w-4 h-4" />
                US/CA: 988 Lifeline
              </a>
              <Link
                to="/resources"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-white border border-calm-300 hover:bg-calm-100 text-calm-800 font-medium text-sm transition-colors"
              >
                <Compass className="w-4 h-4 text-wellness-600" />
                All Helplines & Tools
              </Link>
            </div>
          </div>
          <div className="w-24 h-24 rounded-2xl bg-white border border-wellness-200 flex items-center justify-center text-wellness-600 shadow-md shrink-0">
            <Heart className="w-12 h-12 fill-wellness-100 text-wellness-600" />
          </div>
        </div>
      </section>
    </div>
  );
};
