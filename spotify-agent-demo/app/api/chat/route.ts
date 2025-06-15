export async function POST(req: Request) {
  try {
    const { messages } = await req.json()

    // Get the last message from the user
    const lastMessage = messages[messages.length - 1]

    if (!lastMessage || lastMessage.role !== "user") {
      return new Response("Invalid message format", { status: 400 })
    }

    console.log("Sending message to FastAPI server:", lastMessage.content)

    // Call your FastAPI server at the exact URL you specified
    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        message: lastMessage.content,
      }),
    })

    console.log("FastAPI response status:", response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error("FastAPI error response:", errorText)
      throw new Error(`FastAPI server responded with status: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    console.log("FastAPI response data:", data)

    // Create a streaming response to match the AI SDK format
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        // Send the complete JSON response from your API server without double-escaping
        const fullResponse = JSON.stringify(data)
        const chunk = `0:${JSON.stringify(fullResponse)}\n`
        controller.enqueue(encoder.encode(chunk))
        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "X-Vercel-AI-Data-Stream": "v1",
      },
    })
  } catch (error) {
    console.error("Error calling FastAPI server:", error)

    // Detailed fallback response with connection info
    const fallbackMessage = `🎵 Unable to connect to your Music Concierge Agent at http://127.0.0.1:8000

Please make sure:
1. Your FastAPI server is running
2. The server is accessible at http://127.0.0.1:8000
3. CORS is properly configured

Error: ${error instanceof Error ? error.message : "Unknown error"}`

    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        const chunk = `0:"${fallbackMessage.replace(/"/g, '\\"').replace(/\n/g, "\\n")}"\n`
        controller.enqueue(encoder.encode(chunk))
        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "X-Vercel-AI-Data-Stream": "v1",
      },
    })
  }
}
