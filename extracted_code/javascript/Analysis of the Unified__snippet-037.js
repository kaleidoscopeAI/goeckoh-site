const prompt = this.buildCognitivePrompt(hypothesis, systemContext);
const response = await this.generate(prompt, { temperature: 0.7 });

return this.parseCognitiveResponse(response.response, hypothesis, systemContext);
