"use client";

import React, { useState } from "react";

interface Message {
  role: "user" | "assistant";
  text: string;
  steps?: string[];
}

export default function AgenticChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentSteps, setCurrentSteps] = useState<string[]>([]);

  // Fetches the API URL from environment variables, defaulting to local testing
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setCurrentSteps([]);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });

      if (!response.ok) {
        throw new Error("Backend server encountered an error processing the request.");
      }

      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        text: data.answer,
        steps: data.steps,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error communicating with Agentic RAG API:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "❌ Connection error. Please verify the backend API is live." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Maps backend execution nodes to user-friendly UI badges
  const formatStepPill = (step: string) => {
    const mappings: Record<string, { label: string; color: string }> = {
      retrieve: { label: "🔍 Retrieving Context", color: "bg-blue-100 text-blue-800 border-blue-200" },
      grade_documents: { label: "📐 Grading Evidence", color: "bg-purple-100 text-purple-800 border-purple-200" },
      web_search: { label: "🌐 Web Fallback Activated", color: "bg-amber-100 text-amber-800 border-amber-200" },
      generate: { label: "✍️ Generating Answer", color: "bg-green-100 text-green-800 border-green-200" },
    };
    return mappings[step] || { label: step, color: "bg-gray-100 text-gray-800 border-gray-200" };
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 py-4 px-6 shadow-sm flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Agentic RAG Assistant</h1>
          <p className="text-xs text-slate-500">Self-correcting document intelligence loop powered by LangGraph</p>
        </div>
        <span className="flex h-3 w-3 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
        </span>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl w-full mx-auto">
        {messages.length === 0 && (
          <div className="text-center py-12 text-slate-400 border border-dashed border-slate-200 rounded-xl bg-white shadow-sm mt-8">
            <p className="text-lg font-semibold">Your agent is initialized and ready.</p>
            <p className="text-sm">Ask a question targeted at your uploaded documents to witness the routing control loops.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
            {/* Visual Steps Pills for Assistant Decisions */}
            {msg.role === "assistant" && msg.steps && msg.steps.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2 max-w-[85%]">
                {msg.steps.map((step, sIdx) => {
                  const cfg = formatStepPill(step);
                  return (
                    <span key={sIdx} className={`text-xs font-medium px-2.5 py-1 rounded-full border ${cfg.color}`}>
                      {cfg.label}
                    </span>
                  );
                })}
              </div>
            )}

            {/* Message Bubble */}
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm border ${
              msg.role === "user"
                ? "bg-slate-800 border-slate-900 text-white rounded-br-none"
                : "bg-white border-slate-200 text-slate-800 rounded-bl-none"
            }`}>
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center space-x-2 text-slate-400 text-sm animate-pulse">
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            <span>Agent is routing state transitions...</span>
          </div>
        )}
      </div>

      {/* Input Form */}
      <footer className="bg-white border-t border-slate-200 p-4 shadow-md">
        <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Query your technical documents..."
            disabled={isLoading}
            className="flex-1 border border-slate-300 rounded-xl px-4 py-3 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent disabled:bg-slate-100 disabled:text-slate-400 transition"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-slate-800 hover:bg-slate-700 text-white font-medium text-sm px-5 py-3 rounded-xl shadow transition disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}
