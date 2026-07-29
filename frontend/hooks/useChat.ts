import { useState, useEffect, FormEvent } from "react";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  route?: string;
  confidence?: number;
  executionMetrics?: any;
  sources?: string[];
  sqlQuery?: string;
  isStreaming?: boolean;
  error?: boolean;
};

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");

  useEffect(() => {
    // Initialize session ID from local storage or create a new one
    const storedSession = localStorage.getItem("chat_session_id");
    if (storedSession) {
      setSessionId(storedSession);
    } else {
      const newSession = crypto.randomUUID();
      localStorage.setItem("chat_session_id", newSession);
      setSessionId(newSession);
    }
  }, []);

  const clearSession = () => {
    const newSession = crypto.randomUUID();
    localStorage.setItem("chat_session_id", newSession);
    setSessionId(newSession);
    setMessages([]);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const sendMessage = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    const assistantMessageId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: assistantMessageId, role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/chat/stream`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true"
        },
        body: JSON.stringify({ question: userMessage.content, session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let fullContent = "";
      let metadata: any = null;

      if (reader) {
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; 

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                
                if (parsed.done) {
                  metadata = parsed;
                } else if (parsed.token !== undefined) {
                  fullContent += parsed.token;
                }

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? {
                          ...msg,
                          content: fullContent,
                          route: metadata?.route,
                          confidence: metadata?.confidence,
                          executionMetrics: metadata?.execution_metrics,
                          sources: metadata?.sources,
                          sqlQuery: metadata?.sql_query,
                        }
                      : msg
                  )
                );
              } catch (e) {
                console.error("Error parsing stream chunk:", data, e);
              }
            }
          }
        }
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId ? { ...msg, isStreaming: false } : msg
        )
      );
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                isStreaming: false,
                error: true,
                content: msg.content || "Connection error. Please try again later.",
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return {
    messages,
    input,
    handleInputChange,
    sendMessage,
    isLoading,
    clearSession,
  };
}
