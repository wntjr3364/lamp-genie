"""Bounded prefetches for serializing material records."""
from django.db.models import Count, IntegerField, OuterRef, Prefetch, Subquery, Value
from django.db.models.functions import Coalesce

from ..models import Curve, Device, Process, Property, Sample, Structure
from ..serializers import (
    CHAMPION_POOL_ATTR,
    CHAMPION_POOL_SIZE,
    INLINE_DEVICES_ATTR,
    INLINE_PROCESSES_ATTR,
    INLINE_PROPERTIES_ATTR,
    INLINE_SAMPLES_ATTR,
    INLINE_STRUCTURES_ATTR,
    MAX_INLINE_SUBRESOURCES,
)
from .device_policy import champion_pool_queryset
from .quality import eligible_properties


def _scalar_prefetches(bounds):
    """One bounded row per published scalar: the lowest imported value within the given range.

    The choice is a sliced, ordered, filtered prefetch, so each scalar costs one
    bounded read instead of loading every property of the material.

    The caller's bounds go into the query: the published value has to be the
    lowest value inside the filter, not the lowest value overall.
    """
    from materials.serializers import MaterialSerializer, SCALAR_ROW_ATTR

    prefetches = []
    for key, selector in MaterialSerializer.RANGE_FILTERED_SCALARS.items():
        # Eligible rows before the order-and-slice: the published scalar is
        # never a value with a suspect objection.
        window = eligible_properties(Property.objects.filter(
            created_by__isnull=True, value__isnull=False))
        if selector.get("name"):
            window = window.filter(name__iexact=selector["name"])
        if selector.get("name_contains"):
            window = window.filter(name__icontains=selector["name_contains"])
        low, high = (bounds or {}).get(key, (None, None))
        if low is not None:
            window = window.filter(value__gte=low)
        if high is not None:
            window = window.filter(value__lte=high)
        prefetches.append(Prefetch(
            "properties",
            # (value, id): id breaks value ties so the published number is
            # stable across calls.
            queryset=window.order_by("value", "id")[:1],
            to_attr=SCALAR_ROW_ATTR + key,
        ))
    return prefetches


def _with_material_prefetch(qs, *, bounds=None):
    """Attach everything `MaterialSerializer` reads, in bounded reads.

    A material can own tens of thousands of devices, properties and processes,
    while the serializer inlines at most `MAX_INLINE_SUBRESOURCES` of each, so every
    relation is read through a sliced prefetch. Devices are read twice because the
    two consumers want different orderings:

      * the inline window is ID-ordered, the order the serializer inlines devices
        in; a PCE-ordered fetch would change which devices are inlined;
      * the champion pool is PCE-ordered and provenance-scoped, so the first
        plausible row in it is the published champion (see
        `MaterialSerializer._resolve_champion`, which also handles the case
        where the pool is truncated and none of it survives the screen).

    The counts come from correlated subqueries: two `Count()` annotations over
    different multi-valued relations multiply each other's rows, and
    `distinct=True` fixes the arithmetic only at the cost of the cross product.
    The published scalars come from `_scalar_prefetches`, one row per scalar.
    """
    return (
        qs.prefetch_related(
            Prefetch(
                "structures",
                # Bounded in SQL, as the other relations are.
                queryset=(Structure.objects.order_by("id")
                          .prefetch_related("crystal_sites")
                          [:MAX_INLINE_SUBRESOURCES]),
                to_attr=INLINE_STRUCTURES_ATTR,
            ),
            Prefetch(
                "processes",
                queryset=Process.objects.order_by("id")[:MAX_INLINE_SUBRESOURCES],
                to_attr=INLINE_PROCESSES_ATTR,
            ),
            Prefetch(
                "properties",
                # `quality_issues` travels with the window: PropertySerializer
                # nests them, and each inlined row would otherwise issue its
                # own query.
                queryset=(Property.objects.order_by("id")
                          .prefetch_related("quality_issues")
                          [:MAX_INLINE_SUBRESOURCES]),
                to_attr=INLINE_PROPERTIES_ATTR,
            ),
            *_scalar_prefetches(bounds),
            Prefetch(
                "devices",
                # `quality_issues` travels with the window, as for properties:
                # the inline device serializer nests a device's objections.
                queryset=(Device.objects.order_by("id")
                          .prefetch_related("quality_issues")
                          [:MAX_INLINE_SUBRESOURCES]),
                to_attr=INLINE_DEVICES_ATTR,
            ),
            Prefetch(
                "devices",
                # The policy's own population (suspect devices excluded), so
                # the pool is exactly what `champion_cell` would read.
                queryset=champion_pool_queryset()[:CHAMPION_POOL_SIZE],
                to_attr=CHAMPION_POOL_ATTR,
            ),
            Prefetch(
                "samples",
                # Bounded like the other windows; the inline sample serializer
                # nests each sample's quality issues.
                queryset=(Sample.objects.order_by("id")
                          .prefetch_related("quality_issues")
                          [:MAX_INLINE_SUBRESOURCES]),
                to_attr=INLINE_SAMPLES_ATTR,
            ),
        )
        .annotate(
            # A grouped count subquery returns no row when the material owns
            # none, so the annotation would be NULL, which the serializer reads
            # as "not annotated" and answers with `obj.devices.count()`, one
            # query per material. Coalesce turns it into 0.
            _device_count=Coalesce(
                Subquery(
                    Device.objects.filter(material=OuterRef("pk"))
                    .values("material").annotate(n=Count("pk")).values("n"),
                    output_field=IntegerField(),
                ),
                Value(0),
            ),
            _property_count=Coalesce(
                Subquery(
                    Property.objects.filter(material=OuterRef("pk"))
                    .values("material").annotate(n=Count("pk")).values("n"),
                    output_field=IntegerField(),
                ),
                Value(0),
            ),
            # Same reason as the other two: with the relation bounded,
            # `len(obj.processes.all())` would become one query per material.
            _process_count=Coalesce(
                Subquery(
                    Process.objects.filter(material=OuterRef("pk"))
                    .values("material").annotate(n=Count("pk")).values("n"),
                    output_field=IntegerField(),
                ),
                Value(0),
            ),
            _sample_count=Coalesce(
                Subquery(
                    Sample.objects.filter(material=OuterRef("pk"))
                    .values("material").annotate(n=Count("pk")).values("n"),
                    output_field=IntegerField(),
                ),
                Value(0),
            ),
            # A curve is owned by a material (computed) or by one of its
            # samples (measured), never both: two counts added.
            _curve_count=Coalesce(
                Subquery(
                    Curve.objects.filter(material=OuterRef("pk"))
                    .values("material").annotate(n=Count("pk")).values("n"),
                    output_field=IntegerField(),
                ),
                Value(0),
            ) + Coalesce(
                Subquery(
                    Curve.objects.filter(sample__material=OuterRef("pk"))
                    .values("sample__material").annotate(n=Count("pk")).values("n"),
                    output_field=IntegerField(),
                ),
                Value(0),
            ),
        )
    )
