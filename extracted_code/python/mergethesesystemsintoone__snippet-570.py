"""Builds vocabulary and word counts from the corpus."""
self.vocabulary = set()
self.word_counts = defaultdict(int)
for sentence in self.corpus:
    for word in sentence:
        self.word_counts[word] += 1

# Subsampling of frequent words
if self.subsampling_threshold > 0:
    self.vocabulary = {
        word for word in self.word_counts
        if self.word_counts[word] >= self.min_count and
        (self.word_counts[word] / len(self.corpus)) < self.subsampling_threshold or
        random.random() < (1 - np.sqrt(self.subsampling_threshold / (self.word_counts[word] / len(self.corpus))))
    }
else:
    self.vocabulary = {word for word in self.word_counts if self.word_counts[word] >= self.min_count}

