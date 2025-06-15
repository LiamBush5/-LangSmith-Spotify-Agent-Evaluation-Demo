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
    <div className="group flex items-center px-4 py-2 rounded-md hover:bg-gray-50 transition-colors duration-200">
      {/* Track Number / Play Button */}
      <div className="w-8 flex items-center justify-center mr-4">
        <span className="text-gray-500 text-sm group-hover:hidden">
          {index ? index : "#"}
        </span>
        <Button
          size="sm"
          className="w-8 h-8 p-0 bg-green-500 hover:bg-green-600 text-white rounded-full shadow-md hidden group-hover:flex items-center justify-center"
          onClick={() => window.open(song.spotify_url, "_blank")}
        >
          <Play className="w-3 h-3 ml-0.5" />
        </Button>
      </div>

      {/* Album Art */}
      <div className="w-10 h-10 rounded mr-4 overflow-hidden bg-gray-200 flex-shrink-0">
        {song.album_image_url ? (
          <img
            src={song.album_image_url}
            alt={`${song.album} cover`}
            className="w-full h-full object-cover"
            onError={(e) => {
              const target = e.currentTarget as HTMLImageElement
              const sibling = target.nextElementSibling as HTMLElement
              target.style.display = 'none'
              if (sibling) sibling.style.display = 'flex'
            }}
          />
        ) : null}
        <div className={`w-full h-full bg-gradient-to-br from-green-400 to-green-600 flex items-center justify-center ${song.album_image_url ? 'hidden' : 'flex'}`}>
          <Music className="w-4 h-4 text-white" />
        </div>
      </div>

      {/* Track Info */}
      <div className="flex-1 min-w-0 mr-4">
        <div className="flex flex-col">
          <span className="text-gray-900 font-medium text-sm truncate hover:underline cursor-pointer">
            {song.name}
          </span>
          <span className="text-gray-500 text-xs truncate hover:underline cursor-pointer">
            {song.artist}
          </span>
        </div>
      </div>

      {/* Album Name */}
      <div className="hidden md:block flex-1 min-w-0 mr-4">
        <span className="text-gray-500 text-sm truncate hover:underline cursor-pointer">
          {song.album}
        </span>
      </div>



      {/* Duration */}
      <div className="w-12 text-right ml-4">
        <span className="text-gray-500 text-sm">{song.duration}</span>
      </div>
    </div>
  )
}
