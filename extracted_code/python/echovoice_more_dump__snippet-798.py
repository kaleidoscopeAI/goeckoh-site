import requests
from io import BytesIO
from PIL import Image  # Note: PIL not in libs, but assume via code_execution; fallback numpy Sobel
from scipy.ndimage import sobel  # For edges

class DualEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # For context embed

    async def lookup_images(self, context: str, num=1):
        # Real tool call (simulate async; in prod, use xai API)
        # For now, placeholder; use web_search or search_images
        response = await self.search_images(context)  # Assume tool wrapper
        return response[0]['original'] if response else "https://example.com/default.png"  # First image URL

    def analyze_image(self, url: str):
        # Real download and edge detection
        resp = requests.get(url)
        img = np.array(Image.open(BytesIO(resp.content)).convert('L'))  # Grayscale
        edges_x = sobel(img, axis=0)
        edges_y = sobel(img, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        # Sample points: threshold and random sample
        points = np.argwhere(edges > np.mean(edges) * 2)  # Contour points
        sampled = points[np.random.choice(len(points), min(18000, len(points)), replace=False)] if len(points) > 0 else np.random.rand(18000, 2)
        # Map to 3D: add z=0, or project to sphere
        z = np.random.rand(len(sampled)) * 100 - 50  # Random depth
        points_3d = np.column_stack((sampled[:,1], sampled[:,0], z))  # x,y swap for orientation
        points_3d -= np.mean(points_3d, axis=0)  # Center
        points_3d /= np.max(np.abs(points_3d)) + 1e-8  # Normalize
        points_3d *= 200  # Scale
        return points_3d.tolist()  # List for JS

