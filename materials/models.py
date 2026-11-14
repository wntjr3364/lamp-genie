"""The corpus tables under the names the code uses, and the one model this package adds.

Corpus content is read from the `corpus_*` tables through the classes of `materials.build.models`,
re-exported here under the names the serializers and Genie use; their foreign keys carry the
reverse names the code reads (`properties`, `devices`, `crystal_sites`, `quality_issues`, ...). Choice lists live in `materials.build.vocab`.

`MaterialEmbedding` is this package's own table: one vector per corpus material, written only by the
offline embedding commands (`sync_material_embeddings_batch`, `rebuild_faiss_index`).
"""
from django.db import models

from materials.build.models import (  # noqa: F401 - re-exported names
    BuildRun,
    CrystalSite,
    Curve,
    Device,
    DeviceJVScan,
    DeviceLayer,
    DeviceQualityIssue,
    Material,
    Process,
    Property,
    PropertyQualityIssue,
    Sample,
    SampleQualityIssue,
    Structure,
)


class MaterialEmbedding(models.Model):
    """The stored embedding of one corpus material (RAG search)."""

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    material = models.OneToOneField("corpus.Material", on_delete=models.CASCADE, related_name="embedding")
    embedding = models.JSONField(null=True, blank=True, help_text="OpenAI embedding vector")
    vector_size = models.PositiveIntegerField(default=0)
    text_snapshot = models.TextField(blank=True, help_text="Text the embedding was computed from")
    metadata = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Embedding<{self.material_id}>"
