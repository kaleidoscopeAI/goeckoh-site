609          self.app.route('/mirror/reset_metrics', methods=['POST'])(self.
     mirror_reset_metrics)
610 +        self.app.route('/mirror/stream', methods=['GET'])(self.mirror_s
     tream_events)
611          self.app.before_request(self._enforce_security)
    ⋮
908              return jsonify({"success": False, "error": str(e)}), 500
909 +
910 +    def mirror_stream_events(self):
911 +        """
912 +        Server-Sent Events (SSE) stream of latest transcript/corrected/
     GCL for visualizers.
913 +        Polls the live status every 0.5s and emits JSON payloads.
914 +        """
915 +        if not self.mirror_service:
916 +            return jsonify({"error": "Mirror service not available"}),
     503
917 +
918 +        def event_stream():
919 +            import json
920 +            last_sent = None
921 +            while True:
922 +                st = self.mirror_service.status()
923 +                payload = {
924 +                    "ts": time.time(),
925 +                    "transcript": st.last_transcript,
926 +                    "corrected": st.last_corrected,
927 +                    "gcl": st.gcl,
928 +                    "phenotype_counts": st.phenotype_counts,
929 +                }
930 +                data = json.dumps(payload, ensure_ascii=False)
931 +                if data != last_sent:
932 +                    yield f"data: {data}\n\n"
933 +                    last_sent = data
934 +                time.sleep(0.5)
935 +
936 +        return self.app.response_class(event_stream(), mimetype="text/e
     vent-stream")
937

