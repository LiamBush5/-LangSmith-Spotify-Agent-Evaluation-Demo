"use client"

import { Button } from "@/components/ui/button"
import { Play, Music } from "lucide-react"

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

interface SongCardProps {
  song: Song
  index?: number
}

export function SongCard({ song, index }: SongCardProps) {

  return (
    <div className="group flex items-center px-4 py-3 rounded-lg hover:bg-gradient-to-r hover:from-[#1a1a1a] hover:to-[#242424] transition-all duration-300 hover:shadow-lg hover:shadow-[#1db954]/5 cursor-pointer">
      {/* Track Number / Play Button */}
      <div className="w-8 flex items-center justify-center mr-4">
        <span className="text-white/60 text-sm group-hover:hidden transition-opacity duration-200">
          {index ? index : "#"}
        </span>
        <Button
          size="sm"
          className="w-8 h-8 p-0 bg-gradient-to-r from-[#1db954] to-[#1ed760] hover:from-[#1ed760] hover:to-[#22e55f] text-black rounded-full shadow-lg shadow-[#1db954]/30 hidden group-hover:flex items-center justify-center transition-all duration-300 hover:scale-110"
          onClick={() => window.open(song.spotify_url, "_blank")}
        >
          <Play className="w-3 h-3 ml-0.5" />
        </Button>
      </div>

      {/* Album Art */}
      <div className="w-12 h-12 rounded-lg mr-4 overflow-hidden bg-gradient-to-br from-[#282828] to-[#1a1a1a] flex-shrink-0 group-hover:shadow-lg transition-all duration-300">
        {song.album_image_url ? (
          <img
            src={song.album_image_url}
            alt={`${song.album} cover`}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              const target = e.currentTarget as HTMLImageElement
              const sibling = target.nextElementSibling as HTMLElement
              target.style.display = 'none'
              if (sibling) sibling.style.display = 'flex'
            }}
          />
        ) : null}
        <div className={`w-full h-full bg-gradient-to-br from-[#1db954] to-[#1ed760] flex items-center justify-center ${song.album_image_url ? 'hidden' : 'flex'} group-hover:from-[#1ed760] group-hover:to-[#22e55f] transition-all duration-300`}>
          <Music className="w-5 h-5 text-black" />
        </div>
      </div>

      {/* Track Info */}
      <div className="flex-1 min-w-0 mr-4">
        <div className="flex flex-col">
          <span className="text-white font-medium text-sm truncate hover:text-[#1db954] cursor-pointer transition-colors duration-200 group-hover:font-semibold">
            {song.name}
          </span>
          <span className="text-white/70 text-xs truncate hover:text-white/90 cursor-pointer transition-colors duration-200">
            {song.artist}
          </span>
        </div>
      </div>

      {/* Album Name */}
      <div className="hidden md:block flex-1 min-w-0 mr-4">
        <span
          className="text-white/70 text-sm hover:text-white/90 cursor-pointer transition-colors duration-200 block truncate"
          title={song.album}
        >
          {song.album}
        </span>
      </div>

      {/* Duration */}
      <div className="w-12 text-right ml-4">
        <span className="text-white/70 text-sm group-hover:text-white/90 transition-colors duration-200">{song.duration}</span>
      </div>
    </div>
  )
}
