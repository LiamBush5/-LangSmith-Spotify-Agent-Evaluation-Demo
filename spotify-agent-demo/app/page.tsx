"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Music, ArrowUp, WifiOff, RefreshCw, ExternalLink, Play, Heart, Send } from "lucide-react"
import { SongCard } from "@/components/song-card"
import { Header } from "@/components/header"
import { FeedbackComponent } from "@/components/feedback"

const FASTAPI_URL = "http://127.0.0.1:8000"

interface Song {
  id: string
  name: string
  artist: string
  album: string
  popularity: number
  spotify_url: string
  duration: string
  preview_url?: string | null
  album_image_url?: string | null
  album_image_small?: string | null
  formatted_summary?: string
}

interface ChatMessage {
  role: "user" | "assistant"
  content: string
  songs?: Song[]
  toolsUsed?: string[]
  timestamp?: string
  traceId?: string
}

interface ApiResponse {
  response: string
  tool_trajectory: string[]
  reasoning_steps: Array<{
    tool: string
    input: string
    output: string
  }>
  total_tool_calls: number
  unique_tools_used: string[]
  songs_found: number
  query: string
  trace_id?: string
  success: boolean
  error?: string
}

export default function SpotifyMusicChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState<"checking" | "connected" | "disconnected">("checking")
  const [mounted, setMounted] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    setMounted(true)
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (mounted) {
      checkServerStatus()
      const interval = setInterval(checkServerStatus, 30000)
      return () => clearInterval(interval)
    }
  }, [mounted])

  const checkServerStatus = async () => {
    setServerStatus("checking")
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000)

      const response = await fetch(`${FASTAPI_URL}/health`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal: controller.signal,
      })

      clearTimeout(timeoutId)
      setServerStatus(response.ok ? "connected" : "disconnected")
    } catch (error) {
      setServerStatus("disconnected")
    }
  }

  const sendMessage = async (query: string) => {
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    const userMessage: ChatMessage = {
      role: "user",
      content: query,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput("")

    try {
      const response = await fetch(`${FASTAPI_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ApiResponse & { songs?: Song[] } = await response.json()

      // Use the top-level songs array if present, otherwise fallback to reasoning_steps extraction
      const songs: Song[] = Array.isArray(data.songs) ? data.songs : []

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: data.response,
        songs: songs,
        toolsUsed: data.unique_tools_used,
        timestamp: new Date().toISOString(),
        traceId: data.trace_id
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error("Error sending message:", error)
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please make sure the API server is running and try again.",
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }

  const handleSuggestionClick = (prompt: string) => {
    setInput(prompt)
  }

  // Don't render until mounted to avoid hydration mismatch
  if (!mounted) {
    return null
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-[#121212] via-[#121212] to-[#1a1a1a] relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-20 left-10 w-32 h-32 bg-[#1db954] rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-24 h-24 bg-[#1db954] rounded-full blur-2xl animate-pulse animate-delay-1000"></div>
        <div className="absolute bottom-32 left-1/4 w-40 h-40 bg-[#1db954] rounded-full blur-3xl animate-pulse animate-delay-2000"></div>
        <div className="absolute top-1/2 right-1/3 w-20 h-20 bg-[#1db954] rounded-full blur-2xl animate-pulse animate-delay-3000"></div>
      </div>

      <Header serverStatus={serverStatus} onRefreshStatus={checkServerStatus} />

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 relative z-10">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="text-center py-16">
              {/* Simple welcome icon */}
              <div className="w-20 h-20 mx-auto mb-8 bg-gradient-to-r from-[#1db954] to-[#1ed760] rounded-full flex items-center justify-center">
                <Music className="w-10 h-10 text-black" />
              </div>

              <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-[#1db954] bg-clip-text text-transparent mb-4">
                Welcome to Spotify AI Agent
              </h1>
              <p className="text-white/80 text-lg mb-8 max-w-2xl mx-auto">
                I can create playlists, find artist top songs, search tracks, get recommendations, find concerts, and analyze music for you.
              </p>

              {/* Enhanced suggestion cards with agent capabilities */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                <div
                  className="group bg-gradient-to-br from-[#1a1a1a] to-[#242424] border border-[#282828] rounded-xl p-6 text-left hover:from-[#242424] hover:to-[#2a2a2a] hover:border-[#1db954]/30 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-lg hover:shadow-[#1db954]/10"
                  onClick={() => handleSuggestionClick("Create a smart playlist with Taylor Swift's best songs")}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-2 h-2 bg-[#1db954] rounded-full group-hover:animate-pulse"></div>
                    <p className="text-white/90 text-sm font-medium">"Create a smart playlist with Taylor Swift's best songs"</p>
                  </div>
                  <p className="text-white/50 text-xs">🎵 Smart Playlist Creation</p>
                </div>

                <div
                  className="group bg-gradient-to-br from-[#1a1a1a] to-[#242424] border border-[#282828] rounded-xl p-6 text-left hover:from-[#242424] hover:to-[#2a2a2a] hover:border-[#1db954]/30 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-lg hover:shadow-[#1db954]/10"
                  onClick={() => handleSuggestionClick("What are The Weeknd's top 10 most popular songs?")}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-2 h-2 bg-[#1db954] rounded-full group-hover:animate-pulse"></div>
                    <p className="text-white/90 text-sm font-medium">"What are The Weeknd's top 10 most popular songs?"</p>
                  </div>
                  <p className="text-white/50 text-xs">🎤 Artist Top Songs</p>
                </div>

                <div
                  className="group bg-gradient-to-br from-[#1a1a1a] to-[#242424] border border-[#282828] rounded-xl p-6 text-left hover:from-[#242424] hover:to-[#2a2a2a] hover:border-[#1db954]/30 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-lg hover:shadow-[#1db954]/10"
                  onClick={() => handleSuggestionClick("Find me upbeat electronic dance music for working out")}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-2 h-2 bg-[#1db954] rounded-full group-hover:animate-pulse"></div>
                    <p className="text-white/90 text-sm font-medium">"Find me upbeat electronic dance music for working out"</p>
                  </div>
                  <p className="text-white/50 text-xs">🔍 Track Search & Recommendations</p>
                </div>

                <div
                  className="group bg-gradient-to-br from-[#1a1a1a] to-[#242424] border border-[#282828] rounded-xl p-6 text-left hover:from-[#242424] hover:to-[#2a2a2a] hover:border-[#1db954]/30 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-lg hover:shadow-[#1db954]/10"
                  onClick={() => handleSuggestionClick("Are there any concerts happening near Philadelphia this month?")}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <div className="w-2 h-2 bg-[#1db954] rounded-full group-hover:animate-pulse"></div>
                    <p className="text-white/90 text-sm font-medium">"Are there any concerts happening near Philadelphia this month?"</p>
                  </div>
                  <p className="text-white/50 text-xs">🎪 Concert & Event Search</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <div
                  key={`${message.timestamp}-${index}`}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fadeIn`}
                >
                  <div
                    className={`max-w-[80%] p-4 rounded-xl transition-all duration-300 ${message.role === "user"
                      ? "bg-gradient-to-r from-[#1db954] to-[#1ed760] text-black ml-4 shadow-lg shadow-[#1db954]/20"
                      : "bg-gradient-to-br from-[#1a1a1a] to-[#242424] border border-[#282828] mr-4 hover:border-[#1db954]/30"
                      }`}
                  >
                    {message.role === "user" ? (
                      <p className="whitespace-pre-wrap font-medium">{message.content}</p>
                    ) : (
                      <div className="space-y-4">
                        {message.content && (
                          <div className="text-white whitespace-pre-wrap leading-relaxed">
                            {message.content}
                          </div>
                        )}

                        {message.songs && message.songs.length > 0 && (
                          <div className="mt-6 bg-gradient-to-br from-[#181818] to-[#1a1a1a] rounded-xl border border-[#282828] overflow-hidden shadow-xl">
                            {/* Enhanced Spotify-style header */}
                            <div className="flex items-center px-4 py-3 text-xs font-medium text-white/90 uppercase tracking-wider bg-gradient-to-r from-[#121212] to-[#1a1a1a] border-b border-[#282828]">
                              <div className="w-8 mr-4">#</div>
                              <div className="w-10 mr-4"></div>
                              <div className="flex-1 mr-4">Title</div>
                              <div className="hidden md:block flex-1 mr-4">Album</div>
                              <div className="w-12 text-right ml-4">Time</div>
                            </div>

                            {/* Song list */}
                            <div>
                              {message.songs.map((song: Song, songIndex: number) => (
                                <SongCard key={`message-${index}-song-${song.id || songIndex}-${songIndex}`} song={song} index={songIndex + 1} />
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Add feedback component for assistant messages - AFTER songs */}
                        <FeedbackComponent
                          traceId={message.traceId}
                          onFeedbackSubmit={(feedback) => {
                            console.log(`User gave feedback: ${feedback === 1 ? 'thumbs up' : 'thumbs down'}`)
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start animate-fadeIn">
                  <div className="bg-gradient-to-br from-[#1a1a1a] to-[#242424] border border-[#282828] rounded-xl p-4 mr-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-[#1db954] rounded-full animate-bounce"></div>
                      <div className="w-3 h-3 bg-[#1db954] rounded-full animate-bounce animate-delay-100"></div>
                      <div className="w-3 h-3 bg-[#1db954] rounded-full animate-bounce animate-delay-200"></div>
                      <span className="text-white/70 text-sm ml-2">Analyzing music...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Enhanced Input Area */}
      <div className="bg-gradient-to-r from-[#181818] to-[#1a1a1a] border-t border-[#282828] px-6 py-4 flex-shrink-0 relative z-10">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative">
            <Input
              value={input}
              onChange={handleInputChange}
              placeholder="Ask for songs, artists, or music recommendations..."
              className="w-full h-14 px-6 pr-16 text-base bg-gradient-to-r from-[#242424] to-[#2a2a2a] border-2 border-[#3e3e3e] rounded-full focus:border-[#1db954] focus:ring-0 focus:shadow-lg focus:shadow-[#1db954]/20 placeholder:text-white/60 text-white hover:from-[#2a2a2a] hover:to-[#303030] transition-all duration-300"
              disabled={isLoading || serverStatus !== "connected"}
            />
            <Button
              type="submit"
              size="sm"
              disabled={!input.trim() || isLoading || serverStatus !== "connected"}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 w-10 h-10 p-0 bg-gradient-to-r from-[#1db954] to-[#1ed760] hover:from-[#1ed760] hover:to-[#22e55f] disabled:bg-[#535353] disabled:text-white/40 rounded-full transition-all duration-300 hover:scale-110 hover:shadow-lg hover:shadow-[#1db954]/30"
            >
              <ArrowUp className="w-5 h-5 text-black" />
            </Button>
          </form>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
        .animate-delay-100 {
          animation-delay: 0.1s;
        }
        .animate-delay-200 {
          animation-delay: 0.2s;
        }
        .animate-delay-1000 {
          animation-delay: 1s;
        }
        .animate-delay-2000 {
          animation-delay: 2s;
        }
        .animate-delay-3000 {
          animation-delay: 3s;
        }
      `}</style>
    </div>
  )
}
