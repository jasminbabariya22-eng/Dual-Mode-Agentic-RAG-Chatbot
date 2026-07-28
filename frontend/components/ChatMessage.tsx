import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message } from "../hooks/useChat";
import { User, Bot, AlertCircle, Clock, Database, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export const ChatMessage = ({ message }: { message: Message }) => {
  const isUser = message.role === "user";

  const getRouteIcon = (route?: string) => {
    if (route === "sql") return <Database className="w-3 h-3 mr-1 text-blue-500" />;
    if (route === "rag") return <Search className="w-3 h-3 mr-1 text-green-500" />;
    if (route === "hybrid") return <Bot className="w-3 h-3 mr-1 text-purple-500" />;
    return null;
  };

  return (
    <div className={cn("flex w-full mb-6", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] flex flex-col gap-2 rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "bg-blue-600 text-white rounded-br-none"
            : "bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 border border-gray-100 dark:border-gray-700 rounded-bl-none"
        )}
      >
        <div className="flex items-center gap-2 mb-1">
          {isUser ? (
            <User className="w-4 h-4 opacity-70" />
          ) : (
            <Bot className="w-4 h-4 text-blue-500" />
          )}
          <span className="text-xs font-semibold opacity-70 uppercase tracking-wider">
            {isUser ? "You" : "Agent"}
          </span>
          {message.route && !isUser && (
            <span className="ml-auto flex items-center text-[10px] bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded-full font-medium">
              {getRouteIcon(message.route)}
              {message.route}
            </span>
          )}
        </div>

        <div className="prose prose-sm dark:prose-invert max-w-none break-words">
          {message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          ) : message.isStreaming ? (
            <span className="animate-pulse">Thinking...</span>
          ) : null}
        </div>

        {message.sqlQuery && (
          <details className="mt-2 text-xs border border-gray-200 dark:border-gray-600 rounded-md overflow-hidden bg-gray-50 dark:bg-gray-900">
            <summary className="cursor-pointer px-3 py-2 bg-gray-100 dark:bg-gray-800 font-medium hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
              View Generated SQL
            </summary>
            <div className="p-3 overflow-x-auto font-mono text-[11px] text-blue-600 dark:text-blue-400">
              <pre>{message.sqlQuery}</pre>
            </div>
          </details>
        )}

        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
            <p className="font-semibold mb-1 flex items-center gap-1">
              <Search className="w-3 h-3" /> Sources
            </p>
            <ul className="list-disc pl-4 space-y-1">
              {message.sources.map((source, i) => (
                <li key={i}>{source.replace(/\[|\]/g, "")}</li>
              ))}
            </ul>
          </div>
        )}

        {message.error && (
          <div className="mt-2 flex items-center gap-1 text-xs text-red-500">
            <AlertCircle className="w-3 h-3" />
            <span>Generation encountered an error.</span>
          </div>
        )}

        {message.confidence && (
          <div className="flex items-center gap-3 mt-1 pt-2 text-[10px] text-gray-400 dark:text-gray-500 border-t border-gray-100 dark:border-gray-700">
            <span className="flex items-center gap-1" title="Confidence Score">
              Confidence: {(message.confidence * 100).toFixed(1)}%
            </span>
            {message.executionMetrics?.total_time_ms && (
              <span className="flex items-center gap-1" title="Execution Time">
                <Clock className="w-3 h-3" />
                {message.executionMetrics.total_time_ms.toFixed(0)}ms
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
