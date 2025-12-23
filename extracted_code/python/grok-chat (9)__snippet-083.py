    961 +    def list_routes(self):
    962 +        routes = sorted([r.rule for r in self.app.url_map.iter_rules()])
    963 +        return jsonify({"routes": routes})
    964 +
    965      # ---------------- ASR single-shot endpoint -----------------

