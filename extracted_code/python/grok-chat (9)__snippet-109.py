607          self.app.route('/mirror/stop', methods=['POST'])(self.mirror_st
     op)
608 +        self.app.route('/mirror/validate', methods=['GET'])(self.mirror
     _validate)
609 +        self.app.route('/mirror/reset_metrics', methods=['POST'])(self.
     mirror_reset_metrics)
610          self.app.before_request(self._enforce_security)
    ⋮
887              return jsonify({"success": False, "error": str(e)}), 500
888 +
889 +    def mirror_validate(self):
890 +        """Lightweight validation summary: latency, GCL, drift."""
891 +        if not self.mirror_service:
892 +            return jsonify({"available": False}), 200
893 +        try:
894 +            summary = self.mirror_service.validation_summary()
895 +            return jsonify({"success": True, "summary": summary})
896 +        except Exception as e:
897 +            return jsonify({"success": False, "error": str(e)}), 500
898 +
899 +    def mirror_reset_metrics(self):
900 +        """Reset telemetry for controlled validation runs."""
901 +        if not self.mirror_service:
902 +            return jsonify({"success": False, "error": "Mirror service
     not available"}), 503
903 +        try:
904 +            self.mirror_service.reset_metrics()
905 +            return jsonify({"success": True})
906 +        except Exception as e:
907 +            return jsonify({"success": False, "error": str(e)}), 500
908


