def analyze_text_task(text):
    messages = [LLMMessage(role="user", content=text)]
    llm_result = llm_service.generate(messages)
    return text_node.process(llm_result.content)

