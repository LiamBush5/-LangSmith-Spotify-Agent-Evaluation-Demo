"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Music, ArrowUp, WifiOff, RefreshCw, ExternalLink, Play, Heart, Send } from "lucide-react"
import { SongCard } from "@/components/song-card"

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
  success: boolean
  error?: string
}

export default function SpotifyMusicChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState<"checking" | "connected" | "disconnected">("checking")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    checkServerStatus()
    const interval = setInterval(checkServerStatus, 30000)
    return () => clearInterval(interval)
  }, [])

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
        timestamp: new Date().toISOString()
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

  const formatMessage = (message: ChatMessage) => {
    return (
      <div className={`mb-6 ${message.role === "user" ? "text-right" : "text-left"}`}>
        {message.role === "user" ? (
          <div className="inline-block bg-green-600 text-white px-4 py-2 rounded-lg max-w-xs lg:max-w-md">
            {message.content}
          </div>
        ) : (
          <div className="space-y-4">
            {message.content && (
              <div className="text-gray-800 whitespace-pre-wrap leading-relaxed">
                {message.content}
              </div>
            )}
            {message.toolsUsed && message.toolsUsed.length > 0 && (
              <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                🔧 Tools used: {message.toolsUsed.join(", ")}
              </div>
            )}
            {message.songs && message.songs.length > 0 && (
              <div className="space-y-2">
                {message.songs.map((song: Song, index: number) => (
                  <SongCard key={`song-${song.id || index}`} song={song} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <Music className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-medium text-gray-900">Spotify AI Agent</span>
          </div>

          {/* Server Status */}
          <div className="flex items-center space-x-3">
            <div
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-sm font-medium ${serverStatus === "connected"
                ? "bg-green-100 text-green-700"
                : serverStatus === "disconnected"
                  ? "bg-red-100 text-red-700"
                  : "bg-yellow-100 text-yellow-700"
                }`}
            >
              {serverStatus === "connected" ? (
                <>
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Connected</span>
                </>
              ) : serverStatus === "disconnected" ? (
                <>
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  <span>Offline</span>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                  <span>Checking</span>
                </>
              )}
            </div>
            <Button size="sm" variant="ghost" onClick={checkServerStatus} className="text-gray-500 hover:text-gray-700">
              <RefreshCw className={`w-4 h-4 ${serverStatus === "checking" ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="text-center py-16">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <Music className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900 mb-4">Welcome to Spotify AI Agent</h1>
              <p className="text-gray-600 text-lg mb-8 max-w-2xl mx-auto">
                Ask me about music, get song recommendations, discover new artists, or find the perfect playlist for any mood.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                <div className="bg-white border border-gray-200 rounded-lg p-4 text-left">
                  <p className="text-gray-700 text-sm">"Give me some upbeat pop songs"</p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4 text-left">
                  <p className="text-gray-700 text-sm">"Find me relaxing jazz music"</p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4 text-left">
                  <p className="text-gray-700 text-sm">"What are Green Day's best songs?"</p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4 text-left">
                  <p className="text-gray-700 text-sm">"Songs for a workout playlist"</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"
                    }`}
                >
                  <div
                    className={`max-w-[80%] p-4 rounded-lg ${message.role === "user"
                      ? "bg-green-500 text-white ml-4"
                      : "bg-white border border-gray-200 mr-4"
                      }`}
                  >
                    {message.role === "user" ? (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    ) : (
                      <div className="space-y-4">
                        {message.content && (
                          <div className="text-gray-800 whitespace-pre-wrap leading-relaxed">
                            {message.content}
                          </div>
                        )}
                        {message.songs && message.songs.length > 0 && (
                          <div className="mt-6 bg-white rounded-lg border border-gray-200 overflow-hidden">
                            {/* Spotify-style header */}
                            <div className="flex items-center px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider bg-gray-50 border-b border-gray-200">
                              <div className="w-8 mr-4">#</div>
                              <div className="w-10 mr-4"></div>
                              <div className="flex-1 mr-4">Title</div>
                              <div className="hidden md:block flex-1 mr-4">Album</div>
                              <div className="w-12 text-right ml-4">⏱</div>
                            </div>

                            {/* Song list */}
                            <div>
                              {message.songs.map((song: Song, index: number) => (
                                <SongCard key={`song-${song.id || index}`} song={song} index={index + 1} />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 rounded-lg p-4 mr-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area - Fixed at Bottom */}
      <div className="bg-white border-t border-gray-200 px-6 py-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative">
            <Input
              value={input}
              onChange={handleInputChange}
              placeholder="Ask for songs, artists, or music recommendations..."
              className="w-full h-14 px-6 pr-16 text-base bg-white border-2 border-gray-300 rounded-full focus:border-green-500 focus:ring-0 shadow-sm placeholder:text-gray-400"
              disabled={isLoading || serverStatus !== "connected"}
            />
            <Button
              type="submit"
              size="sm"
              disabled={!input.trim() || isLoading || serverStatus !== "connected"}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 w-10 h-10 p-0 bg-green-500 hover:bg-green-600 disabled:bg-gray-300 rounded-full"
            >
              <ArrowUp className="w-5 h-5 text-white" />
            </Button>
          </form>
        </div>
      </div>
    </div>
  )
}
