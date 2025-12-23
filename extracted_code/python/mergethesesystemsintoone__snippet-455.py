def enhanced_process_software(db, analysis_id):
    process_software(db, analysis_id)
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    for file_path in analysis.decompiled_paths:
        result = file_analyzer.analyze_file(file_path)
        text_result = text_node.process(result["file_path"])
        messages = [LLMMessage(role="user", content=f"Summarize:\n{text_result}")]
        summary = llm_service.generate(messages).content
        pattern_recognizer.recognize_patterns({"cycle": 1, "data": summary})

