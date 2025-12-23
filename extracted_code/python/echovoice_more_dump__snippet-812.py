class PoliteCrawler:
    def __init__(self, user_agents: Optional[List[str]] = None, proxies: Optional[List[dict]] = None, default_delay=1.0, ecf_checker: Optional[Callable] = None):
        self.user_agents = user_agents or [
            "ICA-Bot/1.0 (+https://example.org/ica)",
            "Mozilla/5.0 (compatible; ICA/1.0; +https://example.org/ica)"
        ]
        self.proxies = proxies or []
        self.default_delay = default_delay
        self.last_access = {}
        self.ecf_checker = ecf_checker

    def can_fetch(self, url: str) -> bool:
        from urllib.robotparser import RobotFileParser
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = RobotFileParser()
        try:
            rp.set_url(base + "/robots.txt")
            rp.read()
            return rp.can_fetch("*", url)
        except Exception:
            return False

    def _choose_proxy(self):
        if not self.proxies:
            return None
        return random.choice(self.proxies)

    def fetch(self, url: str, max_retry=3, timeout=15, allow_anonymous_proxy=True):
        if self.ecf_checker:
            allowed, reasons = self.ecf_checker({"type": "network", "cost": 0.05}, {"energy": 1.0})
            if not allowed:
                log.warning("ECF blocked crawl for %s: %s", url, reasons)
                return None
        if not self.can_fetch(url):
            log.info("Robots disallow fetching %s", url)
            return None
        domain = urlparse(url).netloc
        now = time.time()
        last = self.last_access.get(domain, 0)
        wait = max(0.0, self.default_delay - (now - last))
        if wait > 0:
            time.sleep(wait)
        headers = {"User-Agent": random.choice(self.user_agents)}
        proxy = self._choose_proxy() if allow_anonymous_proxy else None
        try:
            for attempt in range(max_retry):
                r = requests.get(url, headers=headers, timeout=timeout, proxies=proxy)
                if r.status_code == 200:
                    self.last_access[domain] = time.time()
                    return r.text
                elif 300 <= r.status_code < 400:
                    continue
                time.sleep(1.0 + attempt * 0.5)
            return None
        except Exception as e:
            log.error("Fetch error: %s", e)
            return None

