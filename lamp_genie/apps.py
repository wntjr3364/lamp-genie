"""Django application configurations of the two apps the agent reads."""
from django.apps import AppConfig


class MaterialsConfig(AppConfig):
    """The `materials` app: the agent, its tools, the embedding table and the commands."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "materials"


class CorpusConfig(AppConfig):
    """The corpus tables (`corpus_*`), defined in `materials.build.models`."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "materials.build"
    label = "corpus"
    verbose_name = "Corpus"
