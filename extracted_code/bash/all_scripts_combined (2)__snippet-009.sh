# Terminal 1: Start backend
uvicorn server:app --reload

# Terminal 2 (optional): Monitor logs
tail -f ~/.echo_companion/logs/*.log
