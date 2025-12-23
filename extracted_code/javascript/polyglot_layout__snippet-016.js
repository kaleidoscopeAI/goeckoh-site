"run": "bash -lc 'if command -v ts-node >/dev/null 2>&1; then ts-node src/ts/agi.ts; elif [ -f bin_ts/agi.js ]; then node bin_ts/agi.js; else node -e "console.log(\"
