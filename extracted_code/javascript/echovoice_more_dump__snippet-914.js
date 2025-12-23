const response = await fetch(ollamaEndpoint, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(requestBody),
});

if (!response.ok) {
  const errorText = await response.text();
  throw new Error(`HTTP error ${response.status}: ${errorText}`);
}

const responseData = await response.json();
// Process the responseData, e.g., responseData.message.content
console.log("Chat Response:", responseData.message.content);
return responseData;

