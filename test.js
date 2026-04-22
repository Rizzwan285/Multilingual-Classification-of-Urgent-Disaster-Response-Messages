const apiKey =  process.env.GROQ_API_KEY; 

fetch("https://api.groq.com/openai/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${apiKey}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    // Groq's exact naming convention for Llama 3.1 8B
    "model": "llama-3.1-8b-instant", 
    "messages": [
      { "role": "user", "content": "Hello! What underlying AI model are you running on?" }
    ]
  })
})
.then(response => response.json())
.then(data => {
    if (data.choices && data.choices.length > 0) {
        console.log("Response:", data.choices[0].message.content);
    } else {
        console.log("Full Data:", data);
    }
})
.catch(error => console.error("Error:", error));