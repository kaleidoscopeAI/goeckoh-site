614          self.app.route('/asr', methods=['POST'])(self.asr_recognize)
615 +        self.app.route('/routes', methods=['GET'])(self.list_routes)
616          self.app.before_request(self._enforce_security)
    ⋮
960
961 +    def list_routes(self):
962 +        routes = sorted([r.rule for r in self.app.url_map.iter_rules()])
963 +        return jsonify({"routes": routes})
964 +
965      # ---------------- ASR single-shot endpoint -----------------

