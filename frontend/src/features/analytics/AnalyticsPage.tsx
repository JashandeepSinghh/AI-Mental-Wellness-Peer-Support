import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
} from 'recharts';
import {
  Activity,
  ShieldCheck,
  Info,
  TrendingUp,
  Lock,
} from 'lucide-react';
import { useConsent } from '../../context/ConsentContext';
import { Link } from 'react-router-dom';

export const AnalyticsPage: React.FC = () => {
  const { analytics } = useConsent();
  const isAnalyticsGranted = analytics?.granted ?? false;

  // Example representative emotion distribution
  const emotionData = [
    { name: 'Joy & Gratitude', count: 14, color: '#14b8a6' },
    { name: 'Stress & Pressure', count: 18, color: '#f59e0b' },
    { name: 'Nervousness', count: 11, color: '#6366f1' },
    { name: 'Sadness', count: 8, color: '#0ea5e9' },
    { name: 'Neutral / Calm', count: 16, color: '#94a3b8' },
    { name: 'Loneliness', count: 6, color: '#ec4899' },
  ];

  // Daily trend data
  const trendData = [
    { day: 'Mon', stressScore: 65, calmScore: 40 },
    { day: 'Tue', stressScore: 70, calmScore: 45 },
    { day: 'Wed', stressScore: 50, calmScore: 60 },
    { day: 'Thu', stressScore: 40, calmScore: 70 },
    { day: 'Fri', stressScore: 55, calmScore: 65 },
    { day: 'Sat', stressScore: 30, calmScore: 85 },
    { day: 'Sun', stressScore: 25, calmScore: 90 },
  ];

  // Coping strategies usage
  const copingData = [
    { strategy: 'Box Breathing', sessions: 8 },
    { strategy: '5-4-3-2-1 Sensory', sessions: 6 },
    { strategy: 'Thought Reframing', sessions: 5 },
    { strategy: 'Body Scan', sessions: 4 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-wellness-100 text-wellness-800 text-xs font-semibold mb-2">
            <Activity className="w-3.5 h-3.5 text-wellness-600" />
            <span>Privacy-First Wellness Insights</span>
          </div>
          <h1 className="text-3xl font-bold text-calm-950 tracking-tight">Emotional Wellness Analytics</h1>
          <p className="text-sm text-calm-600">
            Understand emotional patterns, coping frequency, and resilience trajectories over time.
          </p>
        </div>

        {/* Consent Badge */}
        <div className="flex items-center gap-2">
          {isAnalyticsGranted ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-wellness-50 text-wellness-800 text-xs font-medium border border-wellness-200">
              <ShieldCheck className="w-4 h-4 text-wellness-600" />
              Consent Active (Aggregated)
            </span>
          ) : (
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-amber-50 text-amber-800 text-xs font-medium border border-amber-200">
              <Lock className="w-4 h-4 text-amber-600" />
              <span>Analytics Consent Inactive</span>
              <Link to="/consent" className="underline font-bold text-amber-900 ml-1">
                Enable
              </Link>
            </div>
          )}
        </div>
      </div>

      {/* Notice on Privacy & Demo Data */}
      {!isAnalyticsGranted && (
        <div className="p-4 rounded-2xl bg-calm-50 border border-calm-200 flex items-start gap-3 text-xs sm:text-sm text-calm-600">
          <Info className="w-5 h-5 text-calm-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-calm-900">
              Analytics recording is currently disabled in your privacy settings.
            </p>
            <p className="text-xs">
              Because you have not granted analytics consent, no personal session data is logged to disk. The visualizations below represent illustrative baseline samples to show how the dashboard operates when analytics are enabled.
            </p>
          </div>
        </div>
      )}

      {/* Metric Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-5 rounded-2xl border border-calm-200 shadow-sm">
          <span className="text-xs font-medium text-calm-500 uppercase tracking-wider block mb-1">
            Total Reflections
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-calm-950">24</span>
            <span className="text-xs text-wellness-700 font-semibold flex items-center">
              <TrendingUp className="w-3 h-3 mr-0.5" /> +12% this week
            </span>
          </div>
          <p className="text-[11px] text-calm-400 mt-2">Chat interactions and mood check-ins</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-calm-200 shadow-sm">
          <span className="text-xs font-medium text-calm-500 uppercase tracking-wider block mb-1">
            Predominant State
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-extrabold text-wellness-700">Calm & Reflective</span>
          </div>
          <p className="text-[11px] text-calm-400 mt-2">Evaluated across GoEmotions 28-label model</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-calm-200 shadow-sm">
          <span className="text-xs font-medium text-calm-500 uppercase tracking-wider block mb-1">
            Coping Exercises Used
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-calm-950">14</span>
            <span className="text-xs text-emerald-600 font-semibold">Completed</span>
          </div>
          <p className="text-[11px] text-calm-400 mt-2">Somatic & grounding strategies practiced</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-calm-200 shadow-sm">
          <span className="text-xs font-medium text-calm-500 uppercase tracking-wider block mb-1">
            Safety Escalation Status
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-extrabold text-emerald-600">Stable</span>
          </div>
          <p className="text-[11px] text-calm-400 mt-2">Zero urgent linguistic signals triggered</p>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Chart 1: Emotion Distribution */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-calm-900">Emotion Distribution</h2>
              <p className="text-xs text-calm-500">Aggregated classifications across recent interactions</p>
            </div>
            <span className="text-xs px-2.5 py-1 bg-calm-100 text-calm-700 rounded-md font-medium">
              GoEmotions Model
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} angle={-15} textAnchor="end" />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Emotional Trajectory */}
        <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-calm-900">Stress vs. Calm Trajectory</h2>
              <p className="text-xs text-calm-500">Dreaddit Stress Model vs. VAD Valence Index</p>
            </div>
            <span className="text-xs px-2.5 py-1 bg-wellness-50 text-wellness-800 rounded-md font-medium">
              7-Day Trend
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '12px', fontSize: '12px' }}
                />
                <Line
                  type="monotone"
                  dataKey="stressScore"
                  name="Stress Level"
                  stroke="#f59e0b"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="calmScore"
                  name="Calm Index"
                  stroke="#14b8a6"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Coping Strategies Frequency */}
      <div className="bg-white p-6 rounded-2xl border border-calm-200 shadow-sm space-y-4">
        <div>
          <h2 className="text-base font-bold text-calm-900">Most Utilized Coping Strategies</h2>
          <p className="text-xs text-calm-500">Exercises recommended and accessed through your sessions</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {copingData.map((item) => (
            <div key={item.strategy} className="p-4 rounded-xl bg-calm-50 border border-calm-100 space-y-1">
              <span className="text-xs font-semibold text-calm-800">{item.strategy}</span>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-wellness-700">{item.sessions}</span>
                <span className="text-[11px] text-calm-500">times used</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
