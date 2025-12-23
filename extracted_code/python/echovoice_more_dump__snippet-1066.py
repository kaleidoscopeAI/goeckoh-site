urls_to_visit = [start_url]
visited_urls = set()

while True:
    if not urls_to_visit:
        # If we run out of URLs, start over from a popular site
        urls_to_visit.append("https://www.wikipedia.org/")

    url = urls_to_visit.pop(0)
    if url in visited_urls:
        continue

    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text and inject it into a random node
        text = soup.get_text()
        random_node = np.random.choice(network.nodes)
        random_node.information_buffer.append(text)

        # Find new URLs to visit
        for link in soup.find_all('a'):
            new_url = link.get('href')
            if new_url and new_url.startswith('http'):
                urls_to_visit.append(new_url)

        visited_urls.add(url)

    except Exception as e:
        print(f"Error crawling {url}: {e}")

    time.sleep(1) # Be a good netizen
