class DataWrapper:
    data_type: str  # e.g., "text", "image", "numerical", "audio", "video"
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_data(self) -> Any:
        return self.data

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

class ProcessingUnit:
    def __init__(self, data_type: str):
        self.data_type = data_type

    def process(self, data_wrapper: DataWrapper) -> Any:
        """Processes the data and returns a processed representation."""
        raise NotImplementedError

    def update_vectorizer(self, new_texts: List[str]):
      """Updates the TF-IDF vectorizer with new text data."""
      pass

class TextProcessingUnit(ProcessingUnit):
    def __init__(self):
        super().__init__("text")
        self.vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer

    def process(self, data_wrapper: DataWrapper) -> Dict:
        """Processes text data using TF-IDF vectorization."""
        text = data_wrapper.get_data()
        
        # Fit and transform the text data using the vectorizer
        tfidf_matrix = self.vectorizer.fit_transform([text])

        # Convert to a dense array
        return {"tfidf_vector": tfidf_matrix.toarray()}

    def update_vectorizer(self, new_texts: List[str]):
      """Updates the TF-IDF vectorizer with new text data."""
      if new_texts:
         self.vectorizer.fit(new_texts)

class ImageProcessingUnit(ProcessingUnit):
    def __init__(self):
        super().__init__("image")
        
    def process(self, data_wrapper: DataWrapper) -> Dict:
      """Processes image data. Returns various visual descriptors"""
      image = data_wrapper.get_data()
      
      if isinstance (image,str):
           try:
                image = Image.open(image)
           except Exception as e:
               logger.error(f"Failed to open or locate image: {e}")
               return {}

      if image.mode != 'RGB':
          image = image.convert('RGB')

      # Resize the image to a standard size
      image = image.resize((100, 100))
    
      img_array = np.array(image)
      gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
      # Use edge detection as a basic feature
      sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
      sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

      edges_x = self._convolve(gray, sobel_x)
      edges_y = self._convolve(gray, sobel_y)
      edges = np.sqrt(edges_x**2 + edges_y**2)

      # Simple Thresholding (replace with a more sophisticated method if needed)
      threshold = np.mean(edges) + np.std(edges)  # Example threshold
      binary_edges = (edges > threshold).astype(np.uint8) * 255

      # Find Contours (simplified approach)
      contours = self._find_contours(binary_edges)

      shapes = []
      for cnt in contours:
          approx = self._approximate_polygon(cnt, 0.01 * self._calculate_perimeter(cnt))
          shape_type = {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon", 10: "star"}.get(len(approx), "circle")
          if len(approx) == 4:
             x, y, w, h = cv2.boundingRect(cnt)
             aspect_ratio = float (w) / h
             if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "square"
          shapes.append({'type': shape_type, 'vertices': len(approx), 'contour': cnt.tolist()})
      texture = self._analyze_textures(gray)

      return {
          'type': 'visual_pattern',
           'edges': edges.tolist(),
            'shapes': shapes,
           'texture': texture
           }

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
      """Performs a 2D convolution operation."""
      kernel_size = kernel.shape[0]
      pad = kernel_size // 2
      output = np.zeros_like (image, dtype=float)
      padded_image = np.pad(image, pad, mode='constant')

      for i in range(image.shape[0]):
         for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                output [i, j] = np.sum(region * kernel)
      return output
    
    def _find_contours(self, binary_image: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Placeholder for a contour finding algorithm,
        replace this with more advance and sophisticated approach as you build your ai .
        """
        # Basic simplified edge linking approach for now.
        contours = []
        visited = set()

        def dfs(x, y, contour, depth = 0):
          if (x, y) in visited or not (0 <= x < binary_image.shape[0] and 0 <= y < binary_image.shape[1]) or binary_image [x, y] == 0:
            return
          visited.add ((x,y))
          contour.append ((x,y))
          for dx in [-1, 0, 1]:
              for dy in [-1, 0, 1]:
                 if dx != 0 or dy !=0 :
                  dfs (x + dx, y + dy, contour, depth + 1)

        for i in range (binary_image.shape[0]):
          for j in range (binary_image.shape[1]):
                if binary_image [i, j] == 255 and (i,j) not in visited:
                   contour = []
                   dfs(i, j, contour)
                   if len(contour) >= 3: #minimal viable countor for now. can refine.
                       contours.append(contour)
      return contours

    def _calculate_perimeter(self, contour: List[Tuple[int, int]]) -> float:
      """Calculates the perimeter of a contour."""
      perimeter = 0
      for i in range(len(contour)):
         p1 = np.array (contour[i])
         p2 = np.array (contour[(i + 1) % len (contour)]) # next position with modulus operator if list out of bounds
         perimeter += np.linalg.norm (p1 - p2)
      return perimeter
          
    def _approximate_polygon(self, contour: List[Tuple[int, int]], epsilon: float) -> List [Tuple [int, int]]:
        """
          Approximates a contour using the Douglas-Peucker Algorithm by first finding max perpendicular points and spliting contours
          recursive function until a tolerance point or short length.

        : param: Contour, List [Tuple [int, int] ]: List of int values describing image boundary
         : param : epilson value is a small float for error toloerance
         : retrun: approximated verticies based on curve of data
        """
        if len(contour) <= 3:
            return contour
  # Find the point with the maximum distance from the 2 outer ends
        dmax = 0
        index = 0
        for i in range (1, len(contour) -1 ):
          d = self._perpendicular_distance(contour[i], contour[0], contour[-1])
          if d > dmax:
            index = i
            dmax = d

        # If max distance is greater than epsilon recursively process the next levels otherwise simply grab begin and end
      if dmax > epsilon:
        results1 = self._approximate_polygon(contour[:index +1], epsilon)
        results2 = self._approximate_polygon(contour[index:], epsilon)
          
            # build out resuls based on combining previous recursion of all subsets until lowest common base is reaches which returns an edge.
        results = results1[: -1] + results2[:-1]
        
      else:
         results = [contour[0], contour[-1]] # returning ends once lower bound reachd
      
      return results
        

    def _perpendicular_distance (self, point, start, end):
      x0, y0 = point
      x1, y1 = start
      x2, y2 = end

      if x1 == x2 and y1 == y2:
         return math.hypot (x0 - x1, y0- y1)
      else:
         return abs ((x2-x1) * (y1 -y0) (x1 - x0) * (y2- y1)) / math.hypot(x2- x1, y2-y1)

    def _analyze_textures(self, image: Image.Image) -> List[Dict]:
        """
        Analyze textures using statistical measures on pixel neighborhoods.
        """
        try:
          img_array = np.array(image.convert('L')) # Ensure it's grascale
          patterns = []
    
            # Basic LBP Calculation (Iterates through each neighborhood window on an array)
          for i in range(1, img_array.shape[0] - 1):
            for j in range (1, img_array.shape [1] -1) :
                  neighborhood = img_array [i-1:i+2, j-1:j+2]  # Creates small matrix around pixell
                  patterns.append( {
                  'type' : "texture",
                'mean' : np.mean(neighborhood).item(),
                    'variance': np.var(neighborhood).item(), #  Variance - how pixel varies relative to average intensity 
                    'entropy': self._calculate_entropy (neighborhood).item () # Calculate Shannon's entopy as degree of randomenss within an area

            }  )
      
          return patterns
        except Exception as e:
              print (f"Error in_analyze_textures: {e}")
              return []
     
    def _calculate_entropy(self, region: np.ndarray) -> float:
        """Calculates the entropy of a given region."""
        hist, _ = np.histogram(region.flatten(), bins=np.arange(257)) # Bins set 0 - 257
        probs = hist / np.sum (hist)
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probs]) # Base-2 to provide a log result for bits, sum of probability to calculate entropy as information
        return float(entropy)

class SimpleWord2Vec:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, subsampling_threshold: float = 1e-3, negative_samples: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.subsampling_threshold = subsampling_threshold
        self.negative_samples = negative_samples
        self.corpus = []
        self.vocabulary = set()
        self.word_counts = defaultdict(int)
        self.word_vectors = {}  # Word embeddings (target words)
        self.context_vectors = {} # Context word embeddings

    def train(self, sentences: List[List[str]]):
        """Trains the word embedding model on a list of sentences."""
        self.corpus = sentences
        self._build_vocabulary()
        self._initialize_vectors()
        self._train_model()

    def update(self, sentences: List[List[str]]):
        """Updates the model with new sentences, adding to the vocabulary and retraining."""
        self.corpus.extend(sentences)
        self._build_vocabulary()
        self._train_model()

    def _build_vocabulary(self):
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

    def _initialize_vectors(self):
        """Initializes word vectors randomly."""
        for word in self.vocabulary:
            self.word_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size)
            self.context_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size)

    def _train_model(self):
      """Trains the word embedding model using a simplified negative sampling approach."""
      for sentence in self.corpus:
          for i, target_word in enumerate(sentence):
              if target_word not in self.vocabulary:
                  continue
          
              # Get context words within the window
              context_words = sentence [max(0, i - self.window) :i ] + sentence[i+1 : min (len(sentence), i + self.window +1)]
              for context_word in context_words:
                if context_word not in self.vocabulary:
                  continue

                # Negative sampling
                negative_samples = self._get_negative_samples(context_word)
                  
                # Update vectors
                self._update_vectors(target_word, context_word, negative_samples)

    def _get_negative_samples(self, context_word: str) -> List[str]:
      """Samples negative words for a given context word."""
      not_negative_words = set(context_word)
      negative_samples = []
      while len(negative_samples) < self.negative_samples:
          sample = random.choice(list(self.vocabulary - not_negative_words))
          if sample != context_word:
                negative_samples.append(sample)
      return negative_samples
  
    def _update_vectors (self, target_word: str, context_vector: np.ndarray, label: int, learning_rate: float) :
      """Updates word vectors based on positive and negative samples"""
      try:
         context_vector = self.context_vectors [context_word]
         target_vector = self.word_vectors [target_word]
         score = np.dot(target_vector, context_vector)
         prob = 1 / (1 + np.exp(-score)) # Use Sigmoid to ensure results between 0-1
        
      except KeyError:
          return # Skip updating words not already in map

        # calculate gradient
      gradient = learning_rate * (label - prob) * context_vector # Learning gradient using a defined LR to influence step direction

       # Update word vectors with error gradient and reduce context using error gradient
      self.word_vectors [target_word] += gradient
      context_vector -= learning_rate * (label - prob) * self.word_vectors[target_word] # Modify with a different gradient

    def get_vector(self, word: str) -> Optional [np.ndarray]:
        """Returns the word vector for a given word."""
        return self.word_vectors.get(word)

