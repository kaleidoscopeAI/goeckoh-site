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
    print("--- Insight Synthesis Tool Initialized: Ready to forge new wisdom. ---")

def ingest_super_node_insights(self, sn_id, insight_text, keywords=None):
    """
    Ingests a new insight from a Super Node.
    'insight_text' is the primary data for semantic analysis.
    'keywords' can be predefined tags for explicit bridging.
    Robustness: Validate inputs.
    """
    if not isinstance(sn_id, str) or not sn_id:
        print("  [ERROR] Synthesis Directive: sn_id must be a non-empty string.")
        return
    if not isinstance(insight_text, str) or not insight_text:
        print("  [ERROR] Synthesis Directive: insight_text must be a non-empty string.")
        return
    if keywords is not None and not (isinstance(keywords, list) and all(isinstance(k, str) for k in keywords)):
        print("  [ERROR] Synthesis Directive: keywords must be a list of strings or None.")
        return

    new_insight = {'source_sn': sn_id, 'insight_summary': insight_text, 'keywords': keywords or []}
    try:
        self.insights_df = pd.concat([self.insights_df, pd.DataFrame([new_insight])], ignore_index=True)
        print(f"  Synthesis Directive: Ingested insight from '{sn_id}': '{insight_text[:50]}...'")
    except Exception as e:
        print(f"  [ERROR] Synthesis Directive: Failed to ingest insight from '{sn_id}': {e}")


def _generate_embeddings(self):
    """
    Generates conceptual embeddings for insights (simplified via TF-IDF for CPU).
    Robustness: Handle empty dataframes and vectorization errors.
    """
    if self.insights_df.empty:
        print("  [WARNING] Synthesis Directive: No insights to generate embeddings for.")
        return
    try:
        # Ensure 'insight_summary' column exists and is string type
        self.insights_df['insight_summary'] = self.insights_df['insight_summary'].astype(str)
        # Handle potential empty strings or NaNs in insight_summary
        corpus = self.insights_df['insight_summary'].fillna('').tolist()
        if not any(corpus): # If all summaries are empty
            print("  [WARNING] Synthesis Directive: All insight summaries are empty. Cannot generate embeddings.")
            self.insights_df['embedding'] = [np.array([])] * len(self.insights_df) # Assign empty array
            return

        self.insights_df['embedding'] = list(self.vectorizer.fit_transform(corpus).toarray())
        print("  Synthesis Directive: Generated insight embeddings.")
    except Exception as e:
        print(f"  [ERROR] Synthesis Directive: Error generating embeddings: {e}")
        self.insights_df['embedding'] = [np.array([])] * len(self.insights_df) # Assign empty array on error

def identify_semantic_clusters(self):
    """
    Identifies semantic clusters among insights using clustering (simplified 'Semantic Bridging').
    These clusters represent areas of conceptual convergence.
    Robustness: Handle insufficient data for clustering and clustering errors.
    """
    self._generate_embeddings()
    if self.insights_df.empty or 'embedding' not in self.insights_df or self.insights_df['embedding'].apply(lambda x: x.size == 0).all():
        print("  [WARNING] Synthesis Directive: No valid embeddings to cluster.")
        return

    # Filter out insights with empty embeddings if any were created due to errors
    valid_embeddings_df = self.insights_df[self.insights_df['embedding'].apply(lambda x: x.size > 0)]
    if len(valid_embeddings_df) < self.n_clusters:
        print(f"  [WARNING] Synthesis Directive: Not enough valid insights ({len(valid_embeddings_df)}) for clustering (need at least {self.n_clusters}).")
        self.kmeans_model = None # Reset model if not enough data
        return

    embeddings = np.array(valid_embeddings_df['embedding'].tolist())
    try:
        # MiniBatchKMeans is good for larger datasets on CPU as it processes in batches
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        valid_embeddings_df['cluster'] = self.kmeans_model.fit_predict(embeddings)
        # Merge clusters back to original DataFrame
        self.insights_df = self.insights_df.drop(columns=['cluster'], errors='ignore').merge(
            valid_embeddings_df[['cluster']], left_index=True, right_index=True, how='left'
        )
        self.insights_df['cluster'] = self.insights_df['cluster'].fillna(-1).astype(int) # Assign -1 for unclustered
        print(f"  Synthesis Directive: Identified {self.n_clusters} semantic clusters.")
    except Exception as e:
        print(f"  [ERROR] Synthesis Directive: Error during clustering: {e}")
        self.kmeans_model = None # Reset model on error
        self.insights_df['cluster'] = -1 # Assign -1 to all on error


def articulate_breakthroughs(self):
    """
    Conceptualizes the LLM's role in articulating breakthroughs
    by finding connections between clusters or keywords (simplified 'Breakthrough Discovery Engine').
    Robustness: Handle cases where no clusters or insights are available.
    """
    if self.kmeans_model is None or self.insights_df.empty or 'cluster' not in self.insights_df:
        print("  [WARNING] Synthesis Directive: No clusters or insights to analyze for breakthroughs.")
        return []

    breakthroughs = []
    print("\n--- Synthesis Directive: Articulating Potential Breakthroughs ---")

    for cluster_id in range(self.n_clusters):
        cluster_insights = self.insights_df[self.insights_df['cluster'] == cluster_id]
        if len(cluster_insights) < 2: # Need at least two insights to synthesize a breakthrough
            continue

        # Simulate LLM identifying common themes and novel connections
        all_keywords = [kw for kws in cluster_insights['keywords'] for kw in kws]
        common_keywords = pd.Series(all_keywords).value_counts().index.tolist()
        main_theme = common_keywords[0] if common_keywords else "unspecified theme"

        # "New Math" - Simplified Semantic Bridging: Look for connections across different Super Nodes within a cluster
        source_sns = cluster_insights['source_sn'].unique()
        if len(source_sns) > 1: # Breakthroughs often occur at inter-domain intersections
            breakthrough_text = (
                f"**BREAKTHROUGH ALERT (Cluster {cluster_id}, Theme: '{main_theme}')**: "
                f"A novel intersection has been identified, semantically bridging insights from **{', '.join(source_sns)}**.\n"
                f"  Key insights leading to this: " + " | ".join(cluster_insights['insight_summary'].apply(lambda x: x[:40] + "..."))
            )
            # Simulate a causal connection or novel hypothesis based on keywords and source SNs
            if 'risk' in all_keywords and 'supply chain' in all_keywords and 'Finance_SN' in source_sns:
                breakthrough_text += "\n  **Hypothesis**: Unforeseen supply chain vulnerabilities are directly impacting global financial stability, demanding a new integrated risk assessment model."
            elif 'patient outcome' in all_keywords and 'logistics' in all_keywords and 'Healthcare_SN' in source_sns:
                breakthrough_text += "\n  **Hypothesis**: Optimized real-time medical logistics significantly improves patient recovery rates, suggesting a paradigm shift in healthcare delivery efficiency."

            breakthroughs.append(breakthrough_text)

    if not breakthroughs:
        print("  Synthesis Directive: No significant breakthroughs articulated at this moment.")
    return breakthroughs


