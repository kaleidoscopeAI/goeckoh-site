    587      def setup_flask_app(self):
        ⋮
    613          self.app.route('/mirror/stream', methods=['GET'])(self.mirror_stream_events)
    614 +        self.app.route('/asr', methods=['POST'])(self.asr_recognize)
    615          self.app.before_request(self._enforce_security)

