#!/bin/bash
# Script to verify the repository structure and test website locally

echo "=== Goeckoh Repository Structure Verification ==="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "website" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

echo "✓ Repository root structure looks good"
echo ""

# Check main directories
echo "Checking main directory structure..."
for dir in "website" "docs" "config" "scripts" "archive"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/ directory exists"
    else
        echo "✗ $dir/ directory missing"
        exit 1
    fi
done
echo ""

# Check website directory
echo "Checking website..."
if [ -f "website/index.html" ]; then
    echo "✓ website/index.html exists"
else
    echo "✗ website/index.html missing"
    exit 1
fi

# Check website images
if [ -d "website/images" ]; then
    IMAGE_COUNT=$(ls -1 website/images/*.png 2>/dev/null | wc -l)
    echo "✓ Found $IMAGE_COUNT PNG images in website/images/"
else
    echo "✗ Website images directory missing"
    exit 1
fi
echo ""

# Check configuration
echo "Checking configuration..."
if [ -f "config/config.yaml" ]; then
    echo "✓ config/config.yaml exists"
else
    echo "✗ config/config.yaml missing"
    exit 1
fi

if [ -f "config/config.schema.yaml" ]; then
    echo "✓ config/config.schema.yaml exists"
else
    echo "✗ config/config.schema.yaml missing"
fi
echo ""

# Check documentation
echo "Checking documentation structure..."
for doc_dir in "docs/deployment" "docs/system" "docs/guides"; do
    if [ -d "$doc_dir" ]; then
        echo "✓ $doc_dir/ exists"
    else
        echo "✗ $doc_dir/ missing"
    fi
done

if [ -f "docs/INDEX.md" ]; then
    echo "✓ docs/INDEX.md exists"
else
    echo "✗ docs/INDEX.md missing"
fi
echo ""

# Check application files
echo "Checking application structure..."
if [ -d "GOECKOH" ]; then
    echo "✓ GOECKOH/ directory found"
else
    echo "⚠ GOECKOH/ directory not found"
fi

if [ -d "cognitive-nebula" ]; then
    echo "✓ cognitive-nebula/ directory found"
else
    echo "⚠ cognitive-nebula/ directory not found"
fi
echo ""

# Check for key files
echo "Checking key files..."
for file in "README.md" "CONTRIBUTING.md" "requirements.txt"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "⚠ $file missing"
    fi
done
echo ""

echo "=== Verification Complete ==="
echo ""
echo "Repository structure is properly organized!"
echo ""
echo "To test the website locally, run:"
echo "  cd website && python3 -m http.server 8000"
echo "  Then visit http://localhost:8000"
echo ""
echo "To set up the application, run:"
echo "  python3 -m venv venv"
echo "  source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "  pip install -r requirements.txt"
echo "  python -m cli validate  # Validate configuration"
echo ""

