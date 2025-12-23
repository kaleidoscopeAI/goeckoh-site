goeckoh-site/
├── website/                 # Original goeckoh-site content (marketing website)
│   ├── index.html
│   ├── download.html
│   ├── privacy.html
│   ├── terms.html
│   ├── support.html
│   ├── script.js
│   ├── styles.css
│   └── images/             # Renamed from 'assets' to avoid conflicts
│       ├── hero.png
│       ├── logo.png
│       ├── og-image.png
│       └── ...
│
├── GOECKOH/                # Main application directory
│   ├── frontend/           # React frontend
│   └── goeckoh/           # Python backend
│       ├── audio/
│       ├── heart/
│       ├── persistence/
│       └── ...
│
├── assets/                 # Application assets (models)
│   ├── model_stt/
│   └── model_tts/
│
├── docs/                   # Documentation
├── src/                    # Additional source code
├── tests/                  # Test files
├── rust_core/              # Rust components
│
├── README.md               # Updated comprehensive README
├── .gitignore              # Merged .gitignore
└── ...                     # Build scripts, configs, etc.
