import { useState } from 'react'
import { ThumbsUp, ThumbsDown } from 'lucide-react'

interface FeedbackProps {
    traceId?: string
    onFeedbackSubmit?: (feedback: number) => void
}

export function FeedbackComponent({ traceId, onFeedbackSubmit }: FeedbackProps) {
    const [feedback, setFeedback] = useState<number | null>(null)
    const [isSubmitting, setIsSubmitting] = useState(false)

    const handleFeedback = async (value: number) => {
        if (!traceId || isSubmitting) return

        setIsSubmitting(true)
        setFeedback(value)

        try {
            // Use the correct API endpoint - FastAPI server
            const response = await fetch('http://127.0.0.1:8000/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trace_id: traceId,
                    feedback: value
                })
            })

            if (response.ok) {
                onFeedbackSubmit?.(value)
            }
        } catch (error) {
            console.error('Feedback submission failed:', error)
            setFeedback(null) // Reset on error
        } finally {
            setIsSubmitting(false)
        }
    }

    // Debug: Always show the component but show debug info if no traceId
    return (
        <div className="flex items-center space-x-2 mt-2">
            {!traceId ? (
                <div className="text-xs text-yellow-500">
                    No trace ID - feedback disabled
                </div>
            ) : (
                <>
                    <button
                        onClick={() => handleFeedback(1)}
                        disabled={isSubmitting}
                        className={`p-1 rounded transition-colors ${feedback === 1
                            ? 'text-green-500 bg-green-500/10'
                            : 'text-gray-400 hover:text-green-500'
                            }`}
                    >
                        <ThumbsUp size={16} />
                    </button>
                    <button
                        onClick={() => handleFeedback(0)}
                        disabled={isSubmitting}
                        className={`p-1 rounded transition-colors ${feedback === 0
                            ? 'text-red-500 bg-red-500/10'
                            : 'text-gray-400 hover:text-red-500'
                            }`}
                    >
                        <ThumbsDown size={16} />
                    </button>
                </>
            )}
        </div>
    )
}