"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Clock, Users, ChefHat, Heart, Share, BookOpen } from "lucide-react"
import { useState } from "react"

interface RecipeCardProps {
  title: string
  cookTime: string
  servings: string
  difficulty: string
  ingredients: string[]
  instructions: string[]
  tips?: string
}

export function RecipeCard({
  title,
  cookTime,
  servings,
  difficulty,
  ingredients,
  instructions,
  tips,
}: RecipeCardProps) {
  const [showFullRecipe, setShowFullRecipe] = useState(false)
  const [savedRecipe, setSavedRecipe] = useState(false)

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case "easy":
        return "text-green-400 bg-green-400/20"
      case "medium":
        return "text-yellow-400 bg-yellow-400/20"
      case "hard":
        return "text-red-400 bg-red-400/20"
      default:
        return "text-zinc-400 bg-zinc-400/20"
    }
  }

  return (
    <Card className="bg-zinc-800/50 border-zinc-700/50 p-6 my-4 hover:bg-zinc-800/80 transition-all duration-300 group">
      <div className="flex items-start justify-between mb-6">
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-red-500 rounded-xl flex items-center justify-center shadow-lg">
            <ChefHat className="w-8 h-8 text-white" />
          </div>
          <div>
            <h3 className="text-white font-bold text-xl mb-2">{title}</h3>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-1 text-zinc-300">
                <Clock className="w-4 h-4" />
                <span>{cookTime}</span>
              </div>
              <div className="flex items-center space-x-1 text-zinc-300">
                <Users className="w-4 h-4" />
                <span>{servings} servings</span>
              </div>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(difficulty)}`}>
                {difficulty}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <Button
            size="sm"
            variant="ghost"
            className={`text-zinc-400 hover:text-white w-8 h-8 p-0 ${savedRecipe ? "text-red-400" : ""}`}
            onClick={() => setSavedRecipe(!savedRecipe)}
          >
            <Heart className={`w-4 h-4 ${savedRecipe ? "fill-current" : ""}`} />
          </Button>
          <Button size="sm" variant="ghost" className="text-zinc-400 hover:text-white w-8 h-8 p-0">
            <Share className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Ingredients Preview */}
      <div className="mb-6">
        <h4 className="text-white font-semibold mb-3 flex items-center">
          <span className="w-2 h-2 bg-orange-400 rounded-full mr-2"></span>
          Ingredients ({ingredients.length})
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {ingredients.slice(0, showFullRecipe ? ingredients.length : 6).map((ingredient, index) => (
            <div key={index} className="text-zinc-300 text-sm flex items-center py-1">
              <span className="w-1.5 h-1.5 bg-orange-400 rounded-full mr-3 flex-shrink-0"></span>
              {ingredient}
            </div>
          ))}
        </div>
        {ingredients.length > 6 && !showFullRecipe && (
          <p className="text-zinc-400 text-sm mt-3 ml-5">+{ingredients.length - 6} more ingredients</p>
        )}
      </div>

      {/* Instructions Preview/Full */}
      <div className="mb-6">
        <h4 className="text-white font-semibold mb-3 flex items-center">
          <span className="w-2 h-2 bg-orange-400 rounded-full mr-2"></span>
          Instructions
        </h4>
        {showFullRecipe ? (
          <ol className="space-y-4">
            {instructions.map((step, index) => (
              <li key={index} className="text-zinc-300 text-sm flex">
                <span className="bg-orange-500 text-white text-xs rounded-full w-6 h-6 flex items-center justify-center mr-4 mt-0.5 flex-shrink-0 font-medium">
                  {index + 1}
                </span>
                <span className="leading-relaxed">{step}</span>
              </li>
            ))}
          </ol>
        ) : (
          <div className="text-zinc-300 text-sm flex">
            <span className="bg-orange-500 text-white text-xs rounded-full w-6 h-6 flex items-center justify-center mr-4 mt-0.5 flex-shrink-0 font-medium">
              1
            </span>
            <span className="leading-relaxed">{instructions[0]}...</span>
          </div>
        )}
      </div>

      {/* Tips */}
      {tips && showFullRecipe && (
        <div className="mb-6 p-4 bg-gradient-to-r from-orange-900/20 to-red-900/20 rounded-lg border border-orange-700/30">
          <h4 className="text-orange-400 font-semibold mb-2 text-sm flex items-center">
            <span className="mr-2">💡</span>
            Chef's Tip
          </h4>
          <p className="text-zinc-300 text-sm leading-relaxed">{tips}</p>
        </div>
      )}

      {/* Action Button */}
      <Button
        onClick={() => setShowFullRecipe(!showFullRecipe)}
        className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white font-medium py-3 rounded-lg transition-all duration-200 hover:shadow-lg"
      >
        <BookOpen className="w-4 h-4 mr-2" />
        {showFullRecipe ? "Show Less" : "View Full Recipe"}
      </Button>
    </Card>
  )
}
