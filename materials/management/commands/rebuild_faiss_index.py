"""Build the FAISS index on disk, ahead of the first question.

    python -m lamp_genie index
    python -m lamp_genie index --check     # report, build nothing

The index is derived from the vectors in ``MaterialEmbedding``. The build
reads ~3 GiB out of PostgreSQL and peaks at about 55 GiB of RAM, so it runs
as its own command. The agent runs with ``RAG_ALLOW_RUNTIME_EMBEDDING_BUILD``
off and never builds the index; this command builds it outside that guard.

The signature ``(count, latest updated_at)`` is read through the service's own
code path, so this command cannot write an index the service would consider
current when it is not.

The signature only says whether the index matches the vectors. Whether the
vectors still match their text is a separate question: `build_material_text`
reads the Material, its description, properties and devices, none of which
touch `MaterialEmbedding.updated_at`. `--check` answers it by recomputing the
text and comparing it with `MaterialEmbedding.text_snapshot`, the exact string
that was embedded. A timestamp comparison would over-report (a row can change
without changing the text, e.g. a method label)
and would miss deletions (a deleted property changes the text but leaves no
`updated_at` to compare), so the comparison covers every material rather than
a timestamp-filtered candidate set.

A full scan of the 249,075 materials takes a few minutes. `--sample N` spot-checks
the first N materials instead and never reports a corpus verdict.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Exists, OuterRef

from materials.models import Material, MaterialEmbedding

#: `_EMBED_MATERIAL_IMPORTED` is expressed across the embedding's FK; the
#: Material-relative form of the same rule.
_EMBED_MATERIAL_IMPORTED_MATERIAL = {"created_by__isnull": True}


class Command(BaseCommand):
    help = "Build the FAISS index on disk from the stored embeddings."

    def _stale_since_embedding(self, *, sample=None):
        """Return (checked, stale ids) by comparing the embedded text with the current text.

        `text_snapshot` is the string that was embedded, so recomputing it detects
        every kind of change, including a deleted related row, which leaves no
        timestamp behind.
        """
        from django.db import connection

        from materials.models import MaterialEmbedding
        from materials.services.rag import _build_material_text, with_embed_inputs

        if connection.vendor != "postgresql":
            # The comparison cannot run here, which is not the same as "nothing is stale".
            raise CommandError(
                f"the staleness comparison needs PostgreSQL; this connection "
                f"is {connection.vendor!r}. Refusing to report a corpus "
                f"verdict from a comparison that cannot run.")
        # The same population the index, its signature and `n_emb` describe:
        # non-NULL vectors (`get_or_create` can leave a NULL row) of imported
        # materials.
        from materials.services.rag import _EMBED_MATERIAL_IMPORTED

        ids = list(MaterialEmbedding.objects
                   .filter(embedding__isnull=False, **_EMBED_MATERIAL_IMPORTED)
                   .order_by("material_id")
                   .values_list("material_id", flat=True))
        if sample is not None:
            ids = ids[:sample]

        stale = []
        for start in range(0, len(ids), 2000):
            # The same batch inputs the embedder uses (`with_embed_inputs`), so
            # the recomputed text is the one that would be stored, without three
            # queries per material.
            rows = list(MaterialEmbedding.objects
                        .filter(material_id__in=ids[start:start + 2000])
                        .defer("embedding")
                        .select_related("material"))
            with_embed_inputs([row.material for row in rows])
            for row in rows:
                if _build_material_text(row.material)[:4000] != row.text_snapshot:
                    stale.append(row.material_id)
        return len(ids), stale

    def add_arguments(self, parser):
        parser.add_argument(
            "--check", action="store_true",
            help="Report whether the on-disk index matches the corpus, and "
                 "build nothing.")
        parser.add_argument(
            "--sample", type=int, default=None,
            help="Compare only the first N materials and exit 2. A SPOT "
                 "CHECK: it can find staleness but can never establish its "
                 "absence, so it never exits 0.")

    def handle(self, *args, **opts):
        # `_faiss_index_dir()` reads the setting at call time, as the service
        # does when it writes; the module constant is frozen at import.
        from materials.services.rag import _faiss_index_dir, RAGChatService

        index_dir = _faiss_index_dir()
        svc = RAGChatService()
        signature = svc._current_signature()
        sig_file = Path(index_dir) / "signature.txt"
        on_disk = sig_file.read_text().strip() if sig_file.exists() else ""
        want = "|".join(str(p) for p in signature) if signature else ""

        from materials.services.rag import _EMBED_MATERIAL_IMPORTED

        n_emb = (MaterialEmbedding.objects
                 .exclude(embedding__isnull=True)
                 .filter(**_EMBED_MATERIAL_IMPORTED).count())
        # The same population `n_emb` counts (imported materials only).
        n_mat = Material.objects.filter(**_EMBED_MATERIAL_IMPORTED_MATERIAL).count()
        self.stdout.write("")
        self.stdout.write(f"  embeddings   {n_emb:,} / {n_mat:,} materials")
        self.stdout.write(f"  corpus       {want or '(no embeddings)'}")
        self.stdout.write(f"  on disk      {on_disk or '(no index)'}")

        if not signature:
            self.stderr.write(self.style.ERROR(
                "\nno embeddings: run `python -m lamp_genie embed` first. "
                "An index built now would be empty and would answer every "
                "question with nothing found."))
            sys.exit(1)

        # A material with no embedding row is invisible to every later check:
        # the staleness scan compares stored vectors with their text, and the
        # post-build check compares the index with `n_emb`, so a missing
        # material is absent from both sides, so refuse to build until
        # `sync_material_embeddings_batch` (without `--force`) has filled them.
        #
        # One query counts imported materials with no usable vector; two counts
        # read in separate statements could disagree with a concurrent deletion.
        # "Usable" excludes `[]` as well as NULL, as in
        # `sync_material_embeddings_batch`. `Exists(OuterRef)` is used because a
        # reverse `__isnull=True` is a LEFT JOIN predicate that only matches
        # materials with no related row at all.
        _unusable = (MaterialEmbedding.objects
                     .filter(material=OuterRef("pk"))
                     .exclude(embedding__isnull=True)
                     .exclude(embedding=[]))
        n_missing = (Material.objects
                     .filter(**_EMBED_MATERIAL_IMPORTED_MATERIAL)
                     .exclude(Exists(_unusable))
                     .count())
        if n_missing:
            self.stderr.write(self.style.ERROR(
                f"\nNOT EMBEDDED: {n_missing:,} of {n_mat:,} imported "
                f"materials have no embedding row. An index built now would "
                f"omit them silently: the staleness scan cannot see a material "
                f"with no vector, and the signature it publishes would still "
                f"match the corpus, so the agent would load it.\n"
                f"Run `python -m lamp_genie embed` "
                f"(no --force; it embeds only what is missing), then rebuild."))
            sys.exit(1)

        sample = opts.get("sample")
        if sample is not None and sample <= 0:
            # Comparing zero vectors is not a check.
            raise CommandError("--sample must be a positive number of materials")

        checked, stale_ids = self._stale_since_embedding(sample=sample)
        scope = (f"SPOT CHECK, first {checked:,} of {n_emb:,}"
                 if sample is not None else f"all {checked:,}")
        self.stdout.write(
            f"  compared     {checked:>9,}  vectors against the text they "
            f"describe ({scope})")
        style = self.style.ERROR if stale_ids else self.style.SUCCESS
        self.stdout.write(
            f"  {style('STALE' if stale_ids else 'ok   ')}        "
            f"{len(stale_ids):>9,}  whose EMBEDDED TEXT actually differs")

        if sample is not None:
            # A sample can find staleness but never establish its absence, so it
            # never exits 0.
            self.stderr.write(self.style.WARNING(
                f"\nSPOT CHECK over {checked:,} of {n_emb:,} vectors: "
                f"{len(stale_ids):,} stale found here. This says NOTHING about "
                f"the other {n_emb - checked:,}. Run without --sample for a "
                f"corpus verdict."))
            sys.exit(2)

        # A matching signature is not enough: signature.txt can exist without
        # the index files, and without them the agent has no semantic retrieval.
        index_present = all(
            (Path(index_dir) / name).is_file()
            for name in ("index.faiss", "index.pkl")
        )
        if on_disk == want and not stale_ids and index_present:
            self.stdout.write(self.style.SUCCESS(
                "\nthe index on disk already describes this corpus"))
            return
        if on_disk == want and not stale_ids and not index_present:
            # A failure under `--check` only; the plain invocation builds the
            # missing index (the state of a fresh clone).
            if opts["check"]:
                self.stdout.write(self.style.ERROR(
                    f"\nthe signature in {index_dir} matches the corpus, but "
                    f"the index it describes is NOT THERE (index.faiss / "
                    f"index.pkl missing). Run `python -m lamp_genie index` to build it."))
                sys.exit(1)
            self.stdout.write(self.style.WARNING(
                f"\nthe signature in {index_dir} matches the corpus but the "
                f"index is missing -- building it."))
        if stale_ids and opts["check"]:
            # Under `--check` stale vectors are the verdict; for a build they are
            # a warning, since an index over stale vectors still answers while no
            # index leaves the agent without semantic retrieval.
            # Grouped in the database rather than counted in Python.
            by_source = [
                (r["data_source"], r["n"]) for r in
                Material.objects.filter(id__in=stale_ids)
                .values("data_source").annotate(n=Count("id"))
                .order_by("-n")[:8]
            ]
            for source, n in by_source:
                self.stdout.write(f"      {n:>9,}  {source or '(none)'}")
            self.stderr.write(self.style.ERROR(
                f"\nSTALE: {len(stale_ids):,} vectors describe text the corpus "
                f"no longer holds. Rebuilding the index alone would not fix it: "
                f"the vectors themselves are wrong. Run "
                f"`python -m lamp_genie embed --force --source <name>` for "
                f"the sources above, THEN rebuild. The signature line above "
                f"cannot see this: it is computed over MaterialEmbedding, and "
                f"editing a description does not touch that table."))
            sys.exit(1)
        if opts["check"]:
            self.stderr.write(self.style.ERROR(
                "\nSTALE: the index on disk describes a different corpus."))
            sys.exit(1)

        self.stdout.write("\nbuilding...")
        t = time.time()
        if stale_ids:
            self.stdout.write(self.style.WARNING(
                f"\nWARNING: {len(stale_ids):,} vectors are STALE (their embedded text no "
                f"longer matches the corpus). Building anyway; re-embed those "
                f"sources and rebuild again. `--check` reports the breakdown."))
        # A rebuild is never a cache hit: `_get_vectorstore` returns the
        # class-level store when its signature matches and saves nothing.
        RAGChatService._vector_cache = None
        RAGChatService._vector_cache_signature = None
        # This command is the offline builder. RAG_ALLOW_RUNTIME_EMBEDDING_BUILD
        # keeps a question from assembling the index; it does not apply here.
        from materials.services.rag import offline_embedding_writes
        with offline_embedding_writes():
            store = svc._get_vectorstore(offline_builder=True)
        # The save can fail silently (`_save_vectorstore_to_disk` logs and
        # swallows), so success is judged by what is on disk.
        if not all((Path(index_dir) / n).is_file() for n in ("index.faiss", "index.pkl")):
            self.stdout.write(self.style.ERROR(
                f"\nthe build ran but index.faiss / index.pkl are NOT in {index_dir}; "
                f"the save failed (check disk space and permissions, and the log)."))
            sys.exit(1)
        # Existence is not enough: the save is atomic (staged, `os.replace`d,
        # signature written last), so a failed save leaves the previous three
        # files in place. The signature is the commit marker; it is compared
        # with a freshly computed one, which also catches a corpus that changed
        # while the build ran.
        _after = svc._current_signature()
        _want_now = "|".join(str(p) for p in _after) if _after else ""
        _on_disk_now = sig_file.read_text().strip() if sig_file.exists() else ""
        if not _on_disk_now or _on_disk_now != _want_now:
            self.stdout.write(self.style.ERROR(
                f"\nthe build ran but the signature on disk is "
                f"{_on_disk_now or '(absent)'!r} while the corpus is "
                f"{_want_now or '(no embeddings)'!r}.\n"
                f"Either the save failed silently (check disk space, "
                f"permissions and the log) or the corpus changed while the "
                f"build ran, in which case the index already describes a "
                f"state that is gone. Nothing was published; re-run."))
            sys.exit(1)
        took = time.time() - t

        size = sum(p.stat().st_size for p in Path(index_dir).glob("*")
                   if p.is_file())
        self.stdout.write(self.style.SUCCESS(
            f"\n  {store.index.ntotal:,} vectors in {took / 60:.1f} min\n"
            f"  {size / 2 ** 30:.2f} GiB in {index_dir}\n"
            f"  signature {sig_file.read_text().strip()}"))
        if store.index.ntotal != n_emb:
            self.stderr.write(self.style.ERROR(
                f"\nthe index holds {store.index.ntotal:,} vectors but the "
                f"corpus has {n_emb:,} embeddings; a search would silently "
                f"miss the difference."))
            sys.exit(1)
