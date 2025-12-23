async def main():
    """Production execution example"""
    
    # Configure seed URLs
    seed_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/Cognitive_science"
    ]
    
    # Initialize crawler
    crawler = AutonomousCrawlingEngine(
        seed_urls=seed_urls,
        max_concurrent=8,
        rate_limit=0.5
    )
    
    # Execute crawl
    documents = await crawler.crawl(max_documents=50)
    
    # Display results
    print("\n" + "="*70)
    print("🧠 COGNITIVE AUTONOMOUS WEB CRAWLER - RESULTS")
    print("="*70)
    
    print(f"\n📈 Statistics:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Avg CIAE score: {np.mean([d.ciae_score for d in documents]):.3f}")
    print(f"  System emotional state:")
    print(f"    Valence: {crawler.ciae.system_emotional_state.valence:.3f}")
    print(f"    Arousal: {crawler.ciae.system_emotional_state.arousal:.3f}")
    
    print(f"\n🏆 Top 5 Documents by CIAE Score:")
    top_docs = sorted(documents, key=lambda d: d.ciae_score, reverse=True)[:5]
    for i, doc in enumerate(top_docs, 1):
        print(f"  {i}. [{doc.ciae_score:.3f}] {doc.url[:60]}...")
    
    # Export knowledge graph
    crawler.export_knowledge_graph()
    
    print("\n✅ Crawling session complete!")

