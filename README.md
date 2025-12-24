# Goeckoh Site

Static website for Goeckoh speech mirror system.

## About

This is the GitHub Pages website for the Goeckoh project. The site is served directly as static HTML/CSS/JavaScript without Jekyll processing.

## Structure

- `index.html` - Main landing page
- `download.html` - Download page
- `privacy.html` - Privacy policy
- `terms.html` - Terms of service
- `support.html` - Support information
- `images/` - Image assets
- `website/` - Source files (kept for reference)
- `archive/` - Non-website files (not served by GitHub Pages)

## Deployment

The site uses GitHub Pages and is configured with a `.nojekyll` file to bypass Jekyll processing. All website files are in the root directory for direct serving.

## Local Development

Simply open `index.html` in a web browser or use a local HTTP server:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`
