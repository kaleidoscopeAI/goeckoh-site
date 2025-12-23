class InsightSynthesisTool:
    """
    Facilitates multi-domain AI collaboration, generating novel breakthroughs by fusing insights.
    Acts as the 'Breakthrough Discovery Engine' with a 'Semantic Bridging Architecture'.
    """
    def __init__(self, n_clusters=3, max_features=1000):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("InsightSynthesisTool: n_clusters must be a positive integer.")
        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError("InsightSynthesisTool: max_features must be a positive integer.")

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.kmeans_model = None
        self.n_clusters = n_clusters
        self.insights_df = pd.DataFrame(columns=['source_sn', 'insight_summary', 'keywords', 'embedding'])
        logging.info("--- Insight Synthesis Tool Initialized: Ready to forge new wisdom. ---")

    def ingest_super_node_insights(self, sn_id, insight_text, keywords=None):
        """
        Ingests a new insight from a Super Node.
        'insight_text' is the primary data for semantic analysis.
        'keywords' can be predefined tags for explicit bridging.
        Robustness: Validate inputs.
        """
        if not isinstance(sn_id, str) or not sn_id:
            logging.error("Synthesis Directive: sn_id must be a non-empty string.")
            return
        if not isinstance(insight_text, str) or not insight_text:
            logging.error("Synthesis Directive: insight_text must be a non-empty string.")
            return
        if keywords is not None and not (isinstance(keywords, list) and all(isinstance(k, str) for k in keywords)):
            logging.error("Synthesis Directive: keywords must be a list of strings or None.")
            return

        new_insight = {'source_sn': sn_id, 'insight_summary': insight_text, 'keywords': keywords or []}
        try:
            self.insights_df = pd.concat([self.insights_df, pd.DataFrame([new_insight])], ignore_index=True)
            logging.info(f"Synthesis Directive: Ingested insight from '{sn_id}': '{insight_text[:50]}...'")
        except Exception as e:
            logging.error(f"Synthesis Directive: Failed to ingest insight from '{sn_id}': {e}")

    def _generate_embeddings(self):
        """
        Generates conceptual embeddings for insights (simplified via TF-IDF for CPU).
        Robustness: Handle empty dataframes and vectorization errors.
        """
        if self.insights_df.empty:
            logging.warning("Synthesis Directive: No insights to generate embeddings for.")
            return
        try:
            # Ensure 'insight_summary' column exists and is string type
            self.insights_df['insight_summary'] = self.insights_df['insight_summary'].astype(str)
            # Handle potential empty strings or NaNs in insight_summary
            corpus = self.insights_df['insight_summary'].fillna('').tolist()
            if not any(corpus):  # If all summaries are empty
                logging.warning("Synthesis Directive: All insight summaries are empty. Cannot generate embeddings.")
                self.insights_df['embedding'] = [np.array([])] * len(self.insights_df)  # Assign empty array
                return

            self.insights_df['embedding'] = list(self.vectorizer.fit_transform(corpus).toarray())
            logging.info("Synthesis Directive: Generated insight embeddings.")
        except Exception as e:
            logging.error(f"Synthesis Directive: Error generating

