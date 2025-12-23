class AutonomousCrawlingEngine:
    """Production-ready autonomous web crawler"""
    
    def __init__(self,
                 seed_urls: List[str],
                 max_concurrent: int = 10,
                 rate_limit: float = 1.0,
                 user_agent: str = "KaleidoscopeAI/2.0"):
        
        self.seed_urls = seed_urls
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.user_agent = user_agent
        
        # Core components
        self.pii_redactor = AdvancedPIIRedactor()
        self.embedder = QuantumSemanticEmbedder()
        self.emotional_analyzer = EmotionalContentAnalyzer()
        self.ciae = CognitiveInformationAcquisition()
        
        # Crawling state
        self.visited_urls: Set[str] = set()
        self.url_frontier = []  # Priority queue: (-score, url)
        self.crawled_docs: List[CrawledDocument] = []
        self.per_domain_timestamps: Dict[str, float] = {}
        
        # Initialize frontier
        for url in seed_urls:
            self._add_to_frontier(url, priority=1.0)
    
    def _add_to_frontier(self, url: str, priority: float):
        """Add URL to priority frontier"""
        if url not in self.visited_urls:
            # Use negative priority for max-heap behavior
            self.url_frontier.append((-priority, url))
            self.url_frontier.sort()  # Keep sorted
    
    def _normalize_url(self, url: str, base_url: str) -> Optional[str]:
        """Normalize and validate URL"""
        try:
            full_url = urljoin(base_url, url)
            parsed = urlparse(full_url)
            
            if parsed.scheme in ('http', 'https'):
                return full_url
        except:
            pass
        return None
    
    def _respect_rate_limit(self, domain: str):
        """Enforce per-domain rate limiting"""
        last_access = self.per_domain_timestamps.get(domain, 0)
        elapsed = time.time() - last_access
        
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        self.per_domain_timestamps[domain] = time.time()
    
    async def _fetch_page(self, 
                          session: aiohttp.ClientSession,
                          url: str) -> Optional[Tuple[str, List[str]]]:
        """Fetch and parse a single page"""
        try:
            domain = urlparse(url).netloc
            self._respect_rate_limit(domain)
            
            headers = {'User-Agent': self.user_agent}
            
            async with session.get(url, 
                                  headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=15)) as response:
                
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer']):
                    element.decompose()
                
                # Extract text
                text = soup.get_text(separator=' ')
                lines = (line.strip() for line in text.splitlines())
                cleaned_text = ' '.join(line for line in lines if line)
                cleaned_text = cleaned_text[:10000]  # Limit size
                
                # Extract links
                links = []
                for a_tag in soup.find_all('a', href=True):
                    normalized = self._normalize_url(a_tag['href'], url)
                    if normalized:
                        links.append(normalized)
                
                return cleaned_text, links
                
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def _process_document(self,
                         url: str,
                         content: str,
                         links: List[str]) -> CrawledDocument:
        """Complete document processing pipeline"""
        
        # 1. PII Redaction
        redacted_content, pii_stats = self.pii_redactor.redact(content)
        
        # 2. Emotional Analysis
        emotional_sig = self.emotional_analyzer.analyze(redacted_content)
        
        # 3. Semantic Embedding
        embedding = self.embedder.encode(redacted_content)
        
        # 4. Quantum State Initialization
        quantum_state = QuantumInformationState(
            phase=np.random.random() * 2 * np.pi,
            entanglement_degree=len(links) / 100.0
        )
        
        # 5. Create document
        doc = CrawledDocument(
            url=url,
            content=redacted_content[:1000],  # Store sample
            timestamp=time.time(),
            embedding=embedding,
            quantum_state=quantum_state,
            emotional_sig=emotional_sig,
            discovered_links=links,
            pii_redactions=pii_stats
        )
        
        # 6. Compute CIAE score
        context_embeddings = list(self.ciae.visited_embeddings)
        doc.ciae_score = self.ciae.compute_ciae_score(doc, context_embeddings)
        
        return doc
    
    async def crawl(self, max_documents: int = 100) -> List[CrawledDocument]:
        """Execute autonomous crawling session"""
        
        logger.info(f"🚀 Starting autonomous crawl - target: {max_documents} documents")
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            while self.url_frontier and len(self.crawled_docs) < max_documents:
                # Get highest priority URL
                if not self.url_frontier:
                    break
                    
                _, current_url = self.url_frontier.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)
                
                # Fetch page
                result = await self._fetch_page(session, current_url)
                
                if result is None:
                    continue
                
                content, links = result
                
                if len(content) < 100:  # Too short
                    continue
                
                # Process document
                doc = self._process_document(current_url, content, links)
                self.crawled_docs.append(doc)
                
                # Update system state
                success = doc.ciae_score > 0.4
                self.ciae.update_system_state(doc, success)
                
                # Add discovered links to frontier
                for link in links[:20]:  # Limit links per page
                    if link not in self.visited_urls:
                        # Prioritize based on parent document quality
                        link_priority = doc.ciae_score * 0.8
                        self._add_to_frontier(link, link_priority)
                
                logger.info(f"✅ Crawled ({len(self.crawled_docs)}/{max_documents}): "
                          f"{current_url[:60]}... CIAE={doc.ciae_score:.3f}")
                
                await asyncio.sleep(0.1)  # Brief pause
        
        logger.info(f"🎉 Crawl complete: {len(self.crawled_docs)} documents processed")
        return self.crawled_docs
    
    def export_knowledge_graph(self, filename: str = "knowledge_graph.json"):
        """Export knowledge graph for Kaleidoscope integration"""
        kg_export = {
            'documents': [],
            'system_state': {
                'emotional_state': {
                    'valence': self.ciae.system_emotional_state.valence,
                    'arousal': self.ciae.system_emotional_state.arousal,
                    'coherence': self.ciae.system_emotional_state.coherence
                },
                'total_documents': len(self.crawled_docs),
                'avg_ciae_score': np.mean([d.ciae_score for d in self.crawled_docs])
            }
        }
        
        for doc in self.crawled_docs:
            kg_export['documents'].append({
                'url': doc.url,
                'content_sample': doc.content[:200],
                'ciae_score': doc.ciae_score,
                'emotional_signature': {
                    'valence': doc.emotional_sig.valence,
                    'arousal': doc.emotional_sig.arousal,
                    'coherence': doc.emotional_sig.coherence
                },
                'pii_redactions': doc.pii_redactions,
                'timestamp': doc.timestamp
            })
        
        with open(filename, 'w') as f:
            json.dump(kg_export, f, indent=2)
        
        logger.info(f"📊 Knowledge graph exported to {filename}")

