import wikipediaapi

def crawl_wikipedia(node, topic):
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(topic)
    if page.exists():
        node.resources[f"{topic}_summary"] = page.summary
    else:
        node.logs.append(f"Wikipedia page for {topic} does not exist.")

