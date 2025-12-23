fn new(start_url: &str, max_depth: usize) -> Self {
    let base_domain = Url::parse(start_url)
        .expect("Invalid start url")
        .domain()
        .expect("No domain")
        .to_string();

    let mut to_visit = VecDeque::new();
    to_visit.push_back(Url::parse(start_url).unwrap());

    WebCrawler {
        client: Client::new(),
        to_visit,
        visited: HashSet::new(),
        max_depth,
        base_domain,
    }
}

async fn crawl(&mut self) {
    let mut current_depth = 0;

    while !self.to_visit.is_empty() && current_depth <= self.max_depth {
        let mut futures = FuturesUnordered::new();

        // Schedule requests concurrently with politeness delay
        while let Some(url) = self.to_visit.pop_front() {
            if self.visited.contains(url.as_str()) {
                continue;
            }
            if !self.is_same_domain(&url) {
                continue;
            }

            let client = self.client.clone();
            let url_clone = url.clone();

            futures.push(async move {
                println!("Fetching: {}", url_clone);
                task::sleep(Duration::from_millis(REQUEST_DELAY_MS)).await; // politeness
                client.get(url_clone.as_str()).recv_string().await
            });
        }

        // Process responses and extract next URLs
        while let Some(res) = futures.next().await {
            if let Ok(body) = res {
                // Simple HTML parsing for links
                let links = self.extract_links(&body);
                for link in links {
                    if !self.visited.contains(link.as_str()) {
                        self.to_visit.push_back(link);
                    }
                }
            }
        }

        current_depth += 1;
    }
}

fn extract_links(&self, body: &str) -> Vec<Url> {
    let document = Html::parse_document(body);
    let selector = Selector::parse("a[href]").unwrap();

    document.select(&selector)
        .filter_map(|el| el.value().attr("href"))
        .filter_map(|href| Url::parse(href).or_else(|_| Url::parse(&format!("https://{}{}", self.base_domain, href))).ok())
        .filter(|url| self.is_same_domain(url))
        .collect()
}

fn is_same_domain(&self, url: &Url) -> bool {
    if let Some(domain) = url.domain() {
        domain == self.base_domain
    } else {
        false
    }
}
