try {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.REACT_APP_OPENAI_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a visual cognitive AI that describes and imagines thoughts.' },
        { role: 'user', content: input },
      ],
    }),
  });
  const data = await response.json();
  const thought = data.choices?.[0]?.message?.content || 'The AI is silent...';
  setAiThought(thought);
  setAiResponse(thought);
  return thought;
} catch (err) {
  console.error('AI fetch error:', err);
  setAiThought('Error generating response.');
  return 'Error generating response.';
}
