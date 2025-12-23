logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create dummy config file if it doesn't exist
config_path = 'llm_config.json'
if not os.path.exists(config_path):
     default_config = {
         "classification_model_name": "distilbert-base-uncased-finetuned-sst-2-english",
         "summarization_model_name": "t5-small",
         "conversation_model_name": "microsoft/DialoGPT-medium",
         "spacy_model_name": "en_core_web_sm",
         "device": "auto" # Example: let processor auto-detect
     }
     with open(config_path, 'w') as f:
         json.dump(default_config, f, indent=4)
     print(f"Created default config file: {config_path}")

# Initialize processor (will now load config from file)
try:
    # Note: First run might download models, can take time
    # Set device='cpu' in config if no GPU or CUDA issues
    processor = LLMProcessor(config_file=config_path)

    print("\n--- Testing Text Structure ---")
    text1 = "This is the first sentence. This is the second, slightly longer sentence."
    print(processor.analyze_text_structure(text1))

    print("\n--- Testing NER ---")
    text2 = "Apple Inc. is looking at buying U.K. startup for $1 billion in London."
    print(processor.extract_named_entities(text2))

    print("\n--- Testing Classification ---")
    texts_for_classify = ["This is great!", "This is terrible."]
    print(processor.classify_text(texts_for_classify)) # Expected: Positive, Negative (likely classes 1, 0 for sst-2)

    print("\n--- Testing Summarization ---")
    text_for_summary = ["Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents as of 1 January 2024 in an area of more than 105 square kilometres (41 square miles). Since the 17th century, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, gastronomy and science. For its leading role in the arts and sciences, as well as its early and extensive system of street lighting, in the 19th century, it became known as the City of Light."]
    print(processor.summarize_text(text_for_summary))

    print("\n--- Testing Conversation ---")
    print(f"User: Hello there!")
    print(f"Bot: {processor.conversation('Hello there!')}")
    print(f"User: What is your privacy policy?") # Should trigger hardcoded response
    print(f"Bot: {processor.conversation('What is your privacy policy?')}")
    print(f"User: Tell me about the bible.") # Should trigger hardcoded response
    print(f"Bot: {processor.conversation('Tell me about the bible.')}")


    # print("\n--- Testing Web Crawl ---")
    # Note: Web crawling can be slow and hit external sites. Uncomment carefully.
    # url = "https://spacy.io/" # Example site
    # print(f"Crawling {url} (depth 1)...")
    # print(processor.web_crawl(url, max_depth=1))

except Exception as main_e:
     print(f"An error occurred during example usage: {main_e}")




