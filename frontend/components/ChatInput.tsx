import { useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  input: string;
  isLoading: boolean;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSubmit: (e?: React.FormEvent) => void;
}

export function ChatInput({ input, isLoading, onChange, onSubmit }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <form
      onSubmit={onSubmit}
      className="relative flex items-end w-full max-w-4xl mx-auto bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-2xl shadow-sm focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent transition-all overflow-hidden"
    >
      <textarea
        ref={textareaRef}
        value={input}
        onChange={onChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about policies, orders, or products..."
        className="w-full max-h-[200px] bg-transparent border-0 focus:ring-0 resize-none py-4 pl-5 pr-14 text-sm sm:text-base outline-none dark:text-gray-100 placeholder:text-gray-400"
        rows={1}
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={!input.trim() || isLoading}
        className={cn(
          "absolute right-2 bottom-2 p-2 rounded-xl transition-all flex items-center justify-center",
          !input.trim() || isLoading
            ? "bg-gray-100 text-gray-400 dark:bg-gray-700 dark:text-gray-500 cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
        )}
      >
        <Send className="w-5 h-5 ml-[2px]" />
      </button>
    </form>
  );
}
