import wikipediaapi

def crawl_and_learn(node, topic):
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(topic)

    if page.exists():
        annotations = {"summary": page.summary[:200], "keywords": topic}
        node.learn({topic: annotations})
        node.logs.append(f"Crawled Wikipedia for topic: {topic}")
    else:
        node.logs.append(f"Wikipedia page for topic: {topic} does not exist.")

