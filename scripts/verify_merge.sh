#!/bin/bash
# Script to verify the repository merge and test website locally

echo "=== Goeckoh Repository Merge Verification ==="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "website" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

echo "✓ Repository structure looks good"
echo ""

# Check website directory
echo "Checking website directory..."
if [ -d "website" ] && [ -f "website/index.html" ]; then
    echo "✓ Website directory exists with index.html"
else
    echo "✗ Website directory or index.html missing"
    exit 1
fi

# Check website images
echo "Checking website images..."
if [ -d "website/images" ]; then
    IMAGE_COUNT=$(ls -1 website/images/*.png 2>/dev/null | wc -l)
    echo "✓ Found $IMAGE_COUNT PNG images in website/images/"
else
    echo "✗ Website images directory missing"
    exit 1
fi

# Check application files
echo ""
echo "Checking application structure..."
if [ -d "GOECKOH" ] || [ -d "src" ]; then
    echo "✓ Application directories found"
else
    echo "✗ Application directories missing"
    exit 1
fi

# Check for key documentation
echo ""
echo "Checking documentation..."
for doc in "README.md" "REPOSITORY_MERGE.md"; do
    if [ -f "$doc" ]; then
        echo "✓ $doc exists"
    else
        echo "⚠ $doc missing"
    fi
done

echo ""
echo "=== Verification Complete ==="
echo ""
echo "To test the website locally, run:"
echo "  cd website && python3 -m http.server 8000"
echo "  Then visit http://localhost:8000"
echo ""
echo "To set up the application, run:"
echo "  python3 -m venv venv"
echo "  source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "  pip install -r requirements.txt"
echo ""
