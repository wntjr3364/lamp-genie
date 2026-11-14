"""
Batch-sync material embeddings with the OpenAI API, up to BATCH_SIZE texts per call.

Usage:
    python -m lamp_genie embed
    python -m lamp_genie embed --batch-size 200
    python -m lamp_genie embed --force  # re-embed all
"""
import time
import math

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from openai import OpenAI

from materials.models import Material, MaterialEmbedding
from materials.services.rag import with_embed_inputs


BATCH_SIZE = 500  # texts per API call (OpenAI max is 2048)


def build_material_text(material):
    """Return the text embedded for this material.

    Delegates to `rag._build_material_text`, the single definition shared with
    the runtime staleness check, so stored `text_snapshot` values stay
    comparable. Only imported properties and devices are included. Callers
    processing many materials should prefetch the scoped relations into
    `EMBED_PROPS_ATTR`/`EMBED_DEVICES_ATTR` to avoid two queries per material.
    """
    from materials.services.rag import _build_material_text

    return _build_material_text(material)
class Command(BaseCommand):
    help = "Batch-sync OpenAI embeddings for all materials (fast)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size", type=int, default=BATCH_SIZE,
            help=f"Texts per API call (default: {BATCH_SIZE})",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="Re-generate embeddings even for materials that already have them",
        )
        parser.add_argument(
            "--source", action="append", dest="sources", default=None,
            help="Only process materials from this data_source (repeatable)",
        )
        parser.add_argument("--stale-only", action="store_true",
                            help="Embed only records whose current text differs from text_snapshot, or has no snapshot")

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        if batch_size < 1:
            # `--batch-size 0` would make range(0, total, 0) raise.
            raise CommandError("--batch-size must be a positive integer.")
        force = options["force"]
        stale_only = options.get("stale_only", False)
        if force and stale_only:
            raise CommandError("--force and --stale-only cannot be combined")
        sources = options["sources"]
        model = getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # Embed imported materials only. `rag._build_material_text` reads
        # `imported_properties(material)`, so a material with a `created_by` user
        # (none in the deposit) would be embedded from an empty property list
        # and would become retrievable from the FAISS index the agent searches.
        base_qs = Material.objects.filter(created_by__isnull=True)
        if sources:
            base_qs = base_qs.filter(data_source__in=sources)
            self.stdout.write(f"Filtering to sources: {sources}")

        if stale_only:
            material_ids = []
            candidates = list(base_qs.order_by("pk").values_list("pk", flat=True))
            for start in range(0, len(candidates), 2000):
                mids = candidates[start:start + 2000]
                snapshots = dict(MaterialEmbedding.objects.filter(material_id__in=mids)
                                 .values_list("material_id", "text_snapshot"))
                rows = with_embed_inputs(base_qs.filter(pk__in=mids))
                material_ids.extend(m.pk for m in rows if build_material_text(m)[:4000] != snapshots.get(m.pk))
            material_ids.sort()
        elif force:
            material_ids = list(
                base_qs.values_list("id", flat=True).order_by("id")
            )
        else:
            # Materials with no MaterialEmbedding row or a null/empty embedding.
            # A subquery keeps the id list in the database instead of sending a
            # corpus-sized IN list back from Python.
            material_ids = list(
                base_qs.exclude(
                    id__in=MaterialEmbedding.objects
                    .filter(embedding__isnull=False)
                    .exclude(embedding=[])
                    .values("material_id")
                ).values_list("id", flat=True).order_by("id")
            )

        total = len(material_ids)
        self.stdout.write(f"Materials needing embeddings: {total:,}")
        self.stdout.write(f"Batch size: {batch_size}, Model: {model}")

        if total == 0:
            self.stdout.write(self.style.SUCCESS("All embeddings up to date!"))
            return

        t0 = time.time()
        done = 0
        errors = 0

        for batch_start in range(0, total, batch_size):
            batch_ids = material_ids[batch_start:batch_start + batch_size]

            # Load the inputs of `_build_material_text` for the whole batch
            # (`rag.with_embed_inputs`: three aggregate queries), so the text
            # matches what the runtime recomputes and compares with
            # `text_snapshot`.
            materials = with_embed_inputs(
                Material.objects.filter(id__in=batch_ids).order_by("id"))

            texts = []
            mat_list = []
            for mat in materials:
                texts.append(build_material_text(mat))
                mat_list.append(mat)

            # Batch API call: up to 4 attempts with exponential backoff
            # (1s/2s/4s plus jitter) so a transient 429/5xx does not drop the batch.
            import random as _rnd
            response = None
            last_err = None
            for attempt in range(4):
                try:
                    response = client.embeddings.create(
                        model=model,
                        input=texts,
                    )
                    break
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    # A 400 (bad input) is not transient; do not retry it.
                    if "400" in err_str or "invalid" in err_str:
                        break
                    if attempt < 3:
                        sleep_s = (2 ** attempt) + _rnd.uniform(0, 0.5)
                        self.stderr.write(
                            f"  API error at batch {batch_start} (attempt {attempt+1}/4): "
                            f"{e}; retrying in {sleep_s:.1f}s"
                        )
                        time.sleep(sleep_s)
            if response is None:
                self.stderr.write(
                    f"API error at batch {batch_start} after 4 attempts: {last_err}"
                )
                errors += len(texts)
                continue

            embeddings_to_create = []
            embeddings_to_update = []

            existing_embs = {
                emb.material_id: emb
                for emb in MaterialEmbedding.objects.filter(
                    material_id__in=[m.id for m in mat_list]
                )
            }

            # bulk_update() bypasses save() and does not advance `auto_now`
            # fields. The FAISS cache in rag.py keys validity off
            # (count, MAX(updated_at)), so updated_at is set explicitly to make
            # the signature change after a re-embed.
            from django.utils import timezone
            now = timezone.now()

            if len(response.data) != len(texts):
                # Positional matching is only safe if the response is the same
                # length; a short response would silently attach every vector
                # to the wrong material from that point on.
                raise CommandError(
                    f"embedding response has {len(response.data)} vectors for "
                    f"{len(texts)} texts; refusing to attach them by position.")
            indices = [getattr(data, "index", None) for data in response.data]
            if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(texts))):
                raise CommandError("Embedding response indices must be a complete, unique permutation of the input positions")
            for data in response.data:
                # `data.index` is the API's own position, not the loop's.
                i = getattr(data, "index", None)
                if i is None or not (0 <= i < len(mat_list)):
                    raise CommandError(
                        f"embedding response carries an out-of-range index "
                        f"{i!r}; refusing to attach by position.")
                mat = mat_list[i]
                vector = data.embedding
                if not vector or not all(math.isfinite(v) for v in vector):
                    raise CommandError("Embedding response contains an empty or non-finite vector")
                text_snap = texts[i][:4000]

                if mat.id in existing_embs:
                    emb = existing_embs[mat.id]
                    emb.embedding = vector
                    emb.vector_size = len(vector)
                    emb.text_snapshot = text_snap
                    emb.updated_at = now  # force cache-signature invalidation
                    embeddings_to_update.append(emb)
                else:
                    embeddings_to_create.append(
                        MaterialEmbedding(
                            material_id=mat.id,
                            embedding=vector,
                            vector_size=len(vector),
                            text_snapshot=text_snap,
                        )
                    )

            if embeddings_to_create:
                MaterialEmbedding.objects.bulk_create(
                    embeddings_to_create, batch_size=500
                )
            if embeddings_to_update:
                MaterialEmbedding.objects.bulk_update(
                    embeddings_to_update,
                    ["embedding", "vector_size", "text_snapshot", "updated_at"],
                    batch_size=500,
                )

            done += len(mat_list)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            self.stdout.write(
                f"  [{done:,}/{total:,}] "
                f"{rate:.1f} mat/s, ETA {eta/60:.1f}min"
            )

        elapsed = time.time() - t0
        self.stdout.write("")
        self.stdout.write("=" * 60)
        # Exit non-zero on any failed batch, so a chained
        # `sync && rebuild_faiss_index` does not build an index over stale vectors.
        if errors:
            self.stdout.write(self.style.ERROR(
                f"\n{errors:,} material(s) were NOT embedded."))
            raise CommandError(
                f"{errors:,} of {total:,} materials were not embedded; the "
                f"corpus is unchanged for those rows. Do not rebuild the index "
                f"on this result.")
        self.stdout.write(self.style.SUCCESS("BATCH EMBEDDING SYNC COMPLETE"))
        self.stdout.write(f"  Processed: {done:,}")
        self.stdout.write(f"  Errors: {errors:,}")
        self.stdout.write(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        self.stdout.write(
            f"  Total embeddings: "
            f"{MaterialEmbedding.objects.exclude(embedding__isnull=True).count():,}"
        )
