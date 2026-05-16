from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST


class MetricsService:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.predictions_total = Counter(
            "inference_predictions_total", "Total predictions served",
            ["model_version", "emotion"], registry=self.registry,
        )
        self.errors_total = Counter(
            "inference_errors_total", "Errors per endpoint",
            ["endpoint", "kind"], registry=self.registry,
        )
        self.inference_latency = Histogram(
            "inference_latency_seconds", "Inference latency (preprocess+predict)",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.model_load_seconds = Histogram(
            "model_load_seconds", "Time spent loading a model",
            buckets=(0.5, 1, 2, 5, 10, 30, 60), registry=self.registry,
        )
        self.model_switch_seconds = Histogram(
            "model_switch_seconds", "Time spent switching model version",
            buckets=(0.5, 1, 2, 5, 10, 30, 60), registry=self.registry,
        )
        self.active_model_info = Gauge(
            "active_model_info", "Active model version (1=current)",
            ["model_version"], registry=self.registry,
        )
        self.seeder_requests_total = Counter(
            "seeder_requests_total", "Requests recorded from seeder",
            ["status"], registry=self.registry,
        )
        self.observations_total = Counter(
            "observations_total", "Observations stored", registry=self.registry,
        )
        self.drift_score = Gauge(
            "drift_score", "Simulated drift score [0,1]", registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
