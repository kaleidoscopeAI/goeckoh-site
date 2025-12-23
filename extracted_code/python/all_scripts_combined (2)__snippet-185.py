app = create_app(CONFIG, settings_store=settings_store)

@app.post("/api/backend/shutdown")
def shutdown() -> Any:
    request.environ.get("werkzeug.server.shutdown", lambda: None)()
    return jsonify({"status": "ok"})

return app


