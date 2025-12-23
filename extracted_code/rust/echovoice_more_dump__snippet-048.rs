pub async fn track_performance(&self, metrics: PerformanceMetrics) {
    self.metrics.record(metrics).await;
    if metrics.success_rate < 0.8 {
        self.alert_manager.alert("Performance degradation detected").await;
    }
}
