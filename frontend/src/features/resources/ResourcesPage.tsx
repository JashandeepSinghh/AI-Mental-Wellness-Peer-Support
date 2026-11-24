import React, { useState } from 'react';
import {
  Compass,
  PhoneCall,
  Clock,
  Search,
} from 'lucide-react';
import { WELLNESS_RESOURCES, TRUSTED_CRISIS_CONTACTS } from '../../services/wellnessResources';

export const ResourcesPage: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const categories: { key: string; label: string }[] = [
    { key: 'all', label: 'All Resources' },
    { key: 'stress_management', label: 'Stress Management' },
    { key: 'anxiety_coping', label: 'Anxiety Coping' },
    { key: 'emotional_regulation', label: 'Emotional Regulation' },
    { key: 'mindfulness', label: 'Mindfulness' },
    { key: 'sleep_hygiene', label: 'Sleep Hygiene' },
    { key: 'self_care', label: 'Self-Care' },
    { key: 'academic_stress', label: 'Academic Support' },
    { key: 'social_support', label: 'Social Connectedness' },
  ];

  const filteredResources = WELLNESS_RESOURCES.filter((res) => {
    const matchesCategory = selectedCategory === 'all' || res.category === selectedCategory;
    const matchesQuery =
      searchQuery === '' ||
      res.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      res.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
      res.recommendedFor.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesQuery;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Page Header */}
      <div className="max-w-3xl space-y-3">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-wellness-100 text-wellness-800 text-xs font-semibold">
          <Compass className="w-3.5 h-3.5 text-wellness-600" />
          <span>Evidence-Based Self-Care Repository</span>
        </div>
        <h1 className="text-3xl font-bold text-calm-950 tracking-tight">
          Wellness Resources & Grounding Practices
        </h1>
        <p className="text-sm text-calm-600 leading-relaxed">
          Structured coping strategies, somatic exercises, and mental resilience tools designed to support your day-to-day emotional well-being.
        </p>
      </div>

      {/* Emergency & Crisis Support Banner */}
      <div className="bg-gradient-to-r from-wellness-900 to-wellness-800 rounded-2xl p-6 text-white shadow-md">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <PhoneCall className="w-5 h-5 text-wellness-300" />
              Need Immediate or 24/7 Human Crisis Support?
            </h2>
            <p className="text-xs sm:text-sm text-wellness-100 max-w-2xl leading-relaxed">
              If you or a peer are experiencing thoughts of self-harm, severe distress, or an emergency, free and confidential support is accessible right now.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <a
              href="tel:14416"
              className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-wellness-700 hover:bg-wellness-600 text-white font-semibold text-xs transition-colors shadow-sm"
              title="National Mental Health Helpline (India)"
            >
              <PhoneCall className="w-3.5 h-3.5" />
              India: 14416 (Tele-MANAS)
            </a>
            <a
              href="tel:988"
              className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-wellness-600 hover:bg-wellness-500 text-white font-semibold text-xs transition-colors shadow-sm"
              title="988 Suicide & Crisis Lifeline (US & Canada)"
            >
              <PhoneCall className="w-3.5 h-3.5" />
              US/CA: Call/Text 988
            </a>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-wellness-700/60 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {TRUSTED_CRISIS_CONTACTS.map((contact, idx) => (
            <div key={idx} className="bg-wellness-800/80 p-3 rounded-xl border border-wellness-700 text-xs flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between gap-1 mb-1">
                  <span className="font-bold text-white text-[12px]">{contact.name}</span>
                  {contact.region && (
                    <span className="text-[9px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-wellness-700 text-wellness-200 shrink-0">
                      {contact.region}
                    </span>
                  )}
                </div>
                <span className="text-wellness-300 font-semibold block">{contact.contact}</span>
                <p className="text-[11px] text-wellness-100 mt-1 leading-normal">{contact.description}</p>
              </div>
              <span className="text-[10px] text-emerald-300 font-medium mt-2 block">{contact.availability}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Category Pills */}
        <div className="flex flex-wrap gap-1.5 w-full md:w-auto">
          {categories.map((cat) => (
            <button
              key={cat.key}
              onClick={() => setSelectedCategory(cat.key)}
              className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-colors ${
                selectedCategory === cat.key
                  ? 'bg-wellness-600 text-white shadow-sm'
                  : 'bg-white text-calm-700 hover:bg-calm-100 border border-calm-200'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Search input */}
        <div className="relative w-full md:w-64">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-calm-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search exercises, anxiety..."
            className="w-full pl-9 pr-3 py-2 rounded-xl border border-calm-200 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-wellness-500"
          />
        </div>
      </div>

      {/* Resources Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filteredResources.map((res) => (
          <div
            key={res.id}
            id={res.id}
            className="bg-white rounded-2xl border border-calm-200 p-6 shadow-sm hover:shadow-md transition-shadow space-y-4"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-md bg-wellness-50 text-wellness-800 border border-wellness-100">
                {res.category.replace('_', ' ')}
              </span>
              <span className="flex items-center gap-1 text-xs text-calm-400">
                <Clock className="w-3.5 h-3.5" />
                {res.readTimeMinutes} min exercise
              </span>
            </div>

            <div>
              <h3 className="text-lg font-bold text-calm-900 mb-1.5">{res.title}</h3>
              <p className="text-xs sm:text-sm text-calm-600 leading-relaxed">{res.summary}</p>
            </div>

            {/* Guided steps */}
            <div className="space-y-2 pt-2 border-t border-calm-100">
              <h4 className="text-xs font-semibold text-calm-700 uppercase tracking-wider">
                Step-by-step guidance:
              </h4>
              <ol className="space-y-1.5 text-xs text-calm-700">
                {res.guidanceSteps.map((step, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="w-4 h-4 rounded-full bg-wellness-100 text-wellness-800 text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {idx + 1}
                    </span>
                    <span className="leading-relaxed">{step}</span>
                  </li>
                ))}
              </ol>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-1 pt-2">
              <span className="text-[11px] text-calm-400 font-medium mr-1">Recommended for:</span>
              {res.recommendedFor.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-calm-100 text-calm-600 capitalize"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
