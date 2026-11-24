import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Loader2,
  ShieldAlert,
  Sparkles,
  Heart,
  Activity,
  PhoneCall,
  Lock,
  ChevronDown,
  ChevronUp,
  RotateCcw,
} from 'lucide-react';
import { ChatMessage } from '../../types';
import { api } from '../../services/api';
import { processInferenceResult } from '../../services/responseService';

const generateUUID = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

export const ChatPage: React.FC = () => {
  const [conversationId, setConversationId] = useState<string>(generateUUID);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: "Hello, I'm your MindBridge AI peer wellness companion. Whether you're feeling stressed about exams, dealing with anxiety, feeling disconnected, or just want to vent, I'm here to listen and reflect with you. How are you feeling today?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedAnalysisId, setExpandedAnalysisId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleClearConversation = () => {
    setConversationId(generateUUID());
    setMessages([
      {
        id: 'welcome-' + Date.now(),
        sender: 'assistant',
        text: "Conversation refreshed. I'm here whenever you'd like to talk. What is on your mind?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    setInputText('');
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isLoading) return;

    const userMsgId = 'user-' + Date.now();
    const newMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, newMsg]);
    setInputText('');
    setIsLoading(true);

    try {
      // Connect to the real backend inference pipeline!
      const response = await api.sendChatMessage(trimmed, conversationId);

      // Process analysis through empathetic response & safety escalation engine
      const processed = processInferenceResult(response.analysis);

      const aiMsgId = 'ai-' + Date.now();
      const assistantMsg: ChatMessage = {
        id: aiMsgId,
        sender: 'assistant',
        text: processed.supportiveText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        analysis: response.analysis,
        supportiveResponse: processed.supportiveText,
        suggestedResources: processed.recommendedResources,
        safetyGuidance: processed.safetyGuidance,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      // Safe fallback response if inference is temporarily unavailable
      const fallbackMsg: ChatMessage = {
        id: 'fallback-' + Date.now(),
        sender: 'assistant',
        text: "Thank you for reaching out. While our full emotion-analysis inference engine is momentarily catching its breath, please know that your thoughts and feelings are important. If you are experiencing acute distress or need immediate human support, please don't hesitate to reach out to the 988 Suicide & Crisis Lifeline (call/text 988) or Tele-MANAS (14416 in India).",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, fallbackMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleAnalysis = (msgId: string) => {
    setExpandedAnalysisId(expandedAnalysisId === msgId ? null : msgId);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-6 flex flex-col h-[calc(100vh-140px)]">
      {/* Session Header & Privacy Notice */}
      <div className="bg-white px-4 py-3 rounded-2xl border border-calm-200 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-wellness-100 text-wellness-700 flex items-center justify-center">
            <Heart className="w-4 h-4 fill-wellness-200" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-calm-900 flex items-center gap-1.5">
              Confidential Wellness Chat
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </h1>
            <p className="text-[11px] text-calm-500">Connected to multi-model neural inference</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleClearConversation}
            className="flex items-center gap-1 text-[11px] text-calm-600 hover:text-wellness-700 bg-calm-50 hover:bg-calm-100 px-2.5 py-1 rounded-lg border border-calm-200 transition-colors font-medium"
            title="Start a new conversation"
          >
            <RotateCcw className="w-3 h-3 text-wellness-600" />
            <span>New Chat</span>
          </button>
          <div className="flex items-center gap-1.5 text-[11px] text-calm-500 bg-calm-50 px-2.5 py-1 rounded-full border border-calm-200">
            <Lock className="w-3 h-3 text-wellness-600" />
            <span>In-Memory (Zero Storage)</span>
          </div>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 pb-4">
        {messages.map((msg) => {
          const isUser = msg.sender === 'user';
          const isExpanded = expandedAnalysisId === msg.id;
          const emotionLabels =
            msg.analysis?.modules?.emotion?.confidence_analysis?.selected_labels || [];
          const stressClass = msg.analysis?.modules?.stress?.predicted_class;
          const lonelyClass = msg.analysis?.modules?.loneliness?.predicted_class;

          return (
            <div
              key={msg.id}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-full`}
            >
              {/* Message Bubble */}
              <div
                className={`rounded-2xl px-4 py-3 text-sm leading-relaxed max-w-[88%] sm:max-w-[78%] shadow-sm ${
                  isUser
                    ? 'bg-wellness-600 text-white rounded-br-none'
                    : 'bg-white text-calm-900 border border-calm-200 rounded-bl-none'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.text}</p>
                <span
                  className={`block text-[10px] mt-1.5 text-right ${
                    isUser ? 'text-wellness-200' : 'text-calm-400'
                  }`}
                >
                  {msg.timestamp}
                </span>
              </div>

              {/* Safety Guidance Banner (if urgent or concerning) */}
              {!isUser && msg.safetyGuidance && (
                <div
                  className={`mt-2 p-3.5 rounded-xl border max-w-[88%] sm:max-w-[78%] text-xs leading-relaxed ${
                    msg.safetyGuidance.isUrgent
                      ? 'bg-rose-50 border-rose-200 text-rose-900'
                      : 'bg-amber-50 border-amber-200 text-amber-900'
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-bold mb-1">
                    <ShieldAlert
                      className={`w-4 h-4 ${
                        msg.safetyGuidance.isUrgent ? 'text-rose-600' : 'text-amber-600'
                      }`}
                    />
                    <span>
                      {msg.safetyGuidance.isUrgent ? 'Immediate Crisis Support Available' : 'Supportive Notice'}
                    </span>
                  </div>
                  <p>{msg.safetyGuidance.message}</p>
                  <div className="mt-2.5 space-y-1.5">
                    {msg.safetyGuidance.crisisContacts.map((contact, idx) => {
                      const getHref = () => {
                        if (contact.contact.startsWith('http')) return contact.contact;
                        if (contact.contact.includes('741741')) return 'sms:741741';
                        if (contact.contact.includes('14416')) return 'tel:14416';
                        if (contact.contact.includes('1800-599-0019')) return 'tel:18005990019';
                        if (contact.contact.includes('9999 666 555')) return 'tel:+919999666555';
                        if (contact.contact.includes('988')) return 'tel:988';
                        return '#';
                      };
                      return (
                        <div
                          key={idx}
                          className="p-2 bg-white/90 rounded-lg border border-calm-200/60 flex items-center justify-between gap-2"
                        >
                          <div>
                            <div className="flex items-center gap-1.5">
                              <p className="font-semibold text-calm-900">{contact.name}</p>
                              {contact.region && (
                                <span className="text-[9px] uppercase font-bold px-1.5 py-0.2 rounded bg-calm-100 text-calm-600">
                                  {contact.region}
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-calm-600">{contact.description}</p>
                          </div>
                          <a
                            href={getHref()}
                            target={contact.contact.startsWith('http') ? '_blank' : undefined}
                            rel={contact.contact.startsWith('http') ? 'noopener noreferrer' : undefined}
                            className="shrink-0 px-2.5 py-1 bg-wellness-700 text-white text-[11px] font-semibold rounded-md flex items-center gap-1 hover:bg-wellness-800 transition-colors"
                          >
                            <PhoneCall className="w-3 h-3" />
                            {contact.contact}
                          </a>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recommended Resources Chips */}
              {!isUser && msg.suggestedResources && msg.suggestedResources.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5 max-w-[88%] sm:max-w-[78%]">
                  <span className="text-[11px] text-calm-500 flex items-center gap-1 w-full font-medium">
                    <Sparkles className="w-3 h-3 text-wellness-600" />
                    Recommended coping tools:
                  </span>
                  {msg.suggestedResources.map((res) => (
                    <a
                      key={res.id}
                      href={`/resources#${res.id}`}
                      className="text-xs px-2.5 py-1 rounded-lg bg-wellness-50 text-wellness-800 hover:bg-wellness-100 border border-wellness-200 transition-colors font-medium flex items-center gap-1"
                    >
                      <span>{res.title}</span>
                    </a>
                  ))}
                </div>
              )}

              {/* Expandable Model Signal Inspection */}
              {!isUser && msg.analysis && (
                <div className="mt-1.5">
                  <button
                    onClick={() => toggleAnalysis(msg.id)}
                    className="text-[11px] text-calm-500 hover:text-wellness-700 flex items-center gap-1 font-medium transition-colors"
                  >
                    <Activity className="w-3 h-3 text-wellness-600" />
                    <span>AI Model Insights</span>
                    {isExpanded ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </button>

                  {isExpanded && (
                    <div className="mt-2 p-3 bg-white rounded-xl border border-calm-200 text-xs text-calm-700 max-w-md shadow-sm space-y-2">
                      <div className="flex flex-wrap gap-1.5 items-center">
                        <span className="font-semibold text-calm-800">Emotions:</span>
                        {emotionLabels.length > 0 ? (
                          emotionLabels.map((emo) => (
                            <span
                              key={emo}
                              className="px-2 py-0.5 bg-wellness-100 text-wellness-800 rounded font-medium"
                            >
                              {emo}
                            </span>
                          ))
                        ) : (
                          <span className="text-calm-400 italic">None above threshold</span>
                        )}
                      </div>

                      <div className="grid grid-cols-2 gap-2 pt-1 border-t border-calm-100">
                        <div>
                          <span className="text-calm-500">Stress:</span>{' '}
                          <span className="font-semibold capitalize text-calm-800">
                            {stressClass?.replace('_', ' ') || 'None'}
                          </span>
                        </div>
                        <div>
                          <span className="text-calm-500">Loneliness:</span>{' '}
                          <span className="font-semibold capitalize text-calm-800">
                            {lonelyClass?.replace('_', ' ') || 'None'}
                          </span>
                        </div>
                        <div>
                          <span className="text-calm-500">Risk Signal:</span>{' '}
                          <span className="font-semibold text-calm-800">
                            {msg.analysis?.modules?.risk?.risk_category?.replace(/_/g, ' ') || 'Normal'}
                          </span>
                        </div>
                        <div>
                          <span className="text-calm-500">Sentiment:</span>{' '}
                          <span className="font-semibold capitalize text-calm-800">
                            {msg.analysis?.modules?.sentiment?.predicted_class || 'Neutral'}
                          </span>
                        </div>
                      </div>

                      <p className="text-[10px] text-calm-400 pt-1 leading-normal">
                        Signals generated by NLP models. Linguistic indicators, not clinical diagnoses.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* Loading Spinner Indicator */}
        {isLoading && (
          <div className="flex items-center gap-2 text-calm-500 text-xs pl-2">
            <Loader2 className="w-4 h-4 text-wellness-600 animate-spin" />
            <span>Analyzing message and generating empathetic response...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Message Input Box */}
      <form onSubmit={handleSendMessage} className="mt-auto pt-2">
        <div className="relative rounded-2xl border border-calm-300 bg-white shadow-sm focus-within:ring-2 focus-within:ring-wellness-500 focus-within:border-wellness-500 p-2">
          <textarea
            rows={2}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
              }
            }}
            placeholder="Share how you are feeling... (Press Enter to send, Shift+Enter for newline)"
            className="w-full resize-none border-0 focus:outline-none focus:ring-0 text-sm text-calm-900 placeholder-calm-400 px-2 py-1"
          />

          <div className="flex items-center justify-between border-t border-calm-100 pt-2 px-1 text-xs text-calm-400">
            <span>{inputText.length} / 20000 characters</span>
            <button
              type="submit"
              disabled={isLoading || !inputText.trim()}
              className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-wellness-600 hover:bg-wellness-700 text-white font-medium text-xs disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-all"
            >
              {isLoading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              <span>Send</span>
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
