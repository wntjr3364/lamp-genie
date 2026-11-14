"""Run the agent from Python without setting up a Django project.

The agent reads the database through Django's ORM. `configure` sets Django up from keyword arguments and
environment variables, and the other functions call it on first use, so `import lamp_genie; lamp_genie.ask(...)`
works once the environment is set.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse


def _database(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError(f"DATABASE_URL must be a postgresql:// URL, got {url!r}")
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": unquote(parsed.path.lstrip("/")) or "lamp",
        "USER": unquote(parsed.username or ""),
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": parsed.hostname or "localhost",
        "PORT": str(parsed.port or 5432),
    }


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def configure(
    *,
    database_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    index_dir: Optional[str | os.PathLike] = None,
    chat_model: Optional[str] = None,
    **overrides: Any,
) -> None:
    """Configure the agent. Arguments take precedence over environment variables.

    Call it before any other `lamp_genie` function: once the package is configured, a call with
    arguments raises `RuntimeError` and a call without arguments does nothing.

    Args:
        database_url: PostgreSQL URL of the LAMP database (``DATABASE_URL``).
        openai_api_key: OpenAI API key (``OPENAI_API_KEY``).
        index_dir: Directory of the FAISS index (``LAMP_INDEX_DIR``, default ``./faiss_index``).
        chat_model: Chat model (``OPENAI_CHAT_MODEL``, default ``gpt-4o-mini``).
        **overrides: Any further Django setting, e.g. ``LAMP_AGENT_MAX_ITERATIONS=8``.
    """
    import django
    from django.conf import settings

    if settings.configured:
        if overrides or any(v is not None for v in (database_url, openai_api_key, index_dir, chat_model)):
            raise RuntimeError("lamp_genie.configure() must be called before any other lamp_genie "
                               "function; the package is already configured")
        return

    env = os.environ.get
    values = {
        "SECRET_KEY": env("DJANGO_SECRET_KEY", "lamp-genie"),
        "DEBUG": False,
        "INSTALLED_APPS": [
            "django.contrib.auth",
            "django.contrib.contenttypes",
            "lamp_genie.apps.MaterialsConfig",
            "lamp_genie.apps.CorpusConfig",
        ],
        "DATABASES": {
            "default": _database(database_url or env("DATABASE_URL", "postgresql://localhost:5432/lamp"))
        },
        "DEFAULT_AUTO_FIELD": "django.db.models.BigAutoField",
        "USE_TZ": True,
        "TIME_ZONE": "UTC",
        "BASE_DIR": Path.cwd(),
        "OPENAI_API_KEY": openai_api_key or env("OPENAI_API_KEY", ""),
        "OPENAI_CHAT_MODEL": chat_model or env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "OPENAI_EMBEDDING_MODEL": env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        "OPENAI_CHAT_TEMPERATURE": float(env("OPENAI_CHAT_TEMPERATURE", "0.2")),
        "OPENAI_REQUEST_TIMEOUT": float(env("OPENAI_REQUEST_TIMEOUT", "45")),
        "LAMP_AGENT_MAX_ITERATIONS": _positive_int("LAMP_AGENT_MAX_ITERATIONS", 6),
        "LAMP_AGENT_MAX_EXECUTION_TIME": _positive_int("LAMP_AGENT_MAX_EXECUTION_TIME", 60),
        "FAISS_INDEX_DIR": Path(index_dir or env("LAMP_INDEX_DIR", "faiss_index")).resolve(),
        # A question asked before the index exists is answered without semantic retrieval; it never builds the index.
        "RAG_ALLOW_RUNTIME_EMBEDDING_BUILD": False,
    }
    values.update(overrides)
    settings.configure(**values)
    django.setup()


def _run(command: str, *args: str, **options: Any) -> int:
    """Run a management command and return its exit status (0 on success) instead of raising SystemExit.

    A refusal is raised as `django.core.management.CommandError`.
    """
    from django.core.management import call_command

    try:
        call_command(command, *args, **options)
    except SystemExit as exc:
        return 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1
    return 0


def load(deposit_dir: str | os.PathLike, *, verbosity: int = 1) -> int:
    """Create the tables in an empty database and load the unzipped database archive into them.

    Returns 0; a refusal (for example a directory without MANIFEST.json, or a database that already
    holds rows) raises `CommandError`.
    """
    configure()
    from django.core.management import call_command
    from django.core.management.base import CommandError

    if not (Path(deposit_dir) / "MANIFEST.json").exists():
        raise CommandError(f"{deposit_dir}: no MANIFEST.json; pass the directory of the unzipped "
                           "database archive")
    call_command("migrate", interactive=False, verbosity=verbosity)
    return _run("import_public_snapshot", "--input", str(deposit_dir), verbosity=verbosity)


def ask(question: str, history: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
    """Answer one question.

    Args:
        question: The question, in English or Korean.
        history: Earlier turns as ``{"role": "user" | "assistant", "content": ...}``.

    Returns:
        A dict with the answer text (``answer``), the tool outputs (``tool_results``), the records returned by
        the recommendation and similarity tools and by the lookup of a composition described in words
        (``materials``), tool failures (``tool_errors``), and how the run ended (``termination``).
    """
    configure()
    from materials.services.rag import RAGChatService

    return RAGChatService().chat(question=question, history=history)


def embed_records(*args: str) -> int:
    """Embed every material record that has no stored embedding (OpenAI embedding API). Returns 0."""
    configure()
    return _run("sync_material_embeddings_batch", *args)


def build_index(*args: str) -> int:
    """Build the FAISS index from the stored embeddings; pass ``"--check"`` to verify it instead.

    Returns the exit status: 0 when the index is built or, with ``--check``, matches the database;
    1 when it does not (stale vectors, a different corpus, missing embeddings); 2 for ``--sample``.
    """
    configure()
    return _run("rebuild_faiss_index", *args)
