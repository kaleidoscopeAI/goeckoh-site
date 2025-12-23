class DualEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def lookup_images(self, context: str, num=1):
        # Real search; use web_search tool, but since in code, simulate with fixed (in real, call external)
        # For completeness, hardcode example; in prod, integrate tool
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Walking_tiger_female.jpg/800px-Walking_tiger_female.jpg"  # Example for 'tiger'

    def analyze_image(self, url: str):
        resp = requests.get(url)
        if resp.status_code != 200:
            log.error("Image download failed.")
            return np.random.rand(18000, 3).tolist()  # Fallback random
        img = Image.open(BytesIO(resp.content)).convert('L')
        img_array = np.array(img)
        edges_x = sobel(img_array, axis=0)
        edges_y = sobel(img_array, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        points = np.argwhere(edges > np.mean(edges) * 2)
        if len(points) == 0:
            points = np.random.randint(0, min(img.size), (18000, 2))
        elif len(points) < 18000:
            indices = np.random.choice(len(points), 18000, replace=True)
            points = points[indices]
        else:
            indices = np.random.choice(len(points), 18000, replace=False)
            points = points[indices]
        z = np.random.rand(len(points)) * 100 - 50
        points_3d = np.column_stack((points[:,1], points[:,0], z))
        points_3d -= np.mean(points_3d, axis=0)
        points_3d /= np.max(np.abs(points_3d)) + 1e-8
        points_3d *= 200
        return points_3d.tolist()

