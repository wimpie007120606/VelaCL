# Serving

The production entrypoint is `apps.api.main:app`. It loads registry aliases, batches up to 32 texts, emits structured confidence scores and warnings, supports SSE, exposes health/readiness/Prometheus endpoints, and keeps rollback metadata in the registry.
