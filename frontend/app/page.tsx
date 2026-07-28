"use client";

import { useEffect, useRef } from "react";
import { useChat } from "@/hooks/useChat";
import { ChatMessage } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { Bot, Trash2 } from "lucide-react";

export default function Home() {
  const { messages, input, handleInputChange, sendMessage, isLoading, clearSession } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <main className="flex flex-col h-screen max-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md sticky top-0 z-10 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 dark:text-gray-100 leading-tight">
              Agentic RAG Assistant
            </h1>
            <p className="text-xs text-gray-500 font-medium">Dual-Mode Architecture (SQL + RAG)</p>
          </div>
        </div>
        <button
          onClick={clearSession}
          className="text-gray-400 hover:text-red-500 transition-colors p-2 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20"
          title="Clear Conversation"
        >
          <Trash2 className="w-5 h-5" />
        </button>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 w-full">
        <div className="max-w-4xl mx-auto flex flex-col min-h-full">
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center mb-6">
                <Bot className="w-8 h-8" />
              </div>
              <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">
                How can I help you today?
              </h2>
              <p className="text-gray-500 max-w-md mx-auto mb-8 text-sm">
                Ask me about company policies (RAG), order statistics (Text-to-SQL), or complex queries combining both (Hybrid).
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl text-left">
                {[
                  "What is the laptop warranty period?",
                  "How many pending orders do we have?",
                  "Which products under warranty have pending orders?",
                  "Summarize the leave policy."
                ].map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      handleInputChange({ target: { value: suggestion } } as any);
                    }}
                    className="p-3 text-sm border border-gray-200 dark:border-gray-700 rounded-xl hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-gray-700 dark:text-gray-300 transition-all text-left truncate"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 bg-gradient-to-t from-gray-50 via-gray-50 to-transparent dark:from-gray-900 dark:via-gray-900 p-4 sm:p-6 pb-6 sm:pb-8 pt-0 z-10 w-full">
        <ChatInput 
          input={input} 
          isLoading={isLoading} 
          onChange={handleInputChange} 
          onSubmit={sendMessage} 
        />
        <div className="text-center mt-3">
          <span className="text-[10px] text-gray-400 font-medium">
            AI can make mistakes. Verify important information.
          </span>
        </div>
      </div>
    </main>
  );
}
