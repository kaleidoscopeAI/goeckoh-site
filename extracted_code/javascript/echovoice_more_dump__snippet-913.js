const response = await fetch(ollamaEndpoint, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(requestBody),
});

if (!response.ok) {
  // Handle HTTP errors (e.g., 404, 500)
  const errorText = await response.text();
  throw new Error(`HTTP error ${response.status}: ${errorText}`);
}

const responseData = await response.json();
// Process the responseData, e.g., responseData.response contains the generated text
console.log("Generated Response:", responseData.response);
return responseData;

