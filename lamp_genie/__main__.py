"""Command line: python -m lamp_genie {load,embed,index,ask} ..."""
import argparse
import sys

from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import CommandError

from . import api


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lamp_genie", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    load = commands.add_parser("load", help="create the tables in an empty database and load the database archive")
    load.add_argument("deposit_dir", help="the unzipped archive (MANIFEST.json, SHA256SUMS, *.csv.gz)")

    embed = commands.add_parser("embed", help="embed the material records (OpenAI embedding API)")
    embed.add_argument("--batch-size", type=int, help="texts per API request (default 500)")
    embed.add_argument("--force", action="store_true", help="embed every record again")
    embed.add_argument("--source", action="append", default=[], help="only records of this source (repeatable)")
    embed.add_argument("--stale-only", action="store_true",
                       help="only records whose text differs from the text that was embedded")

    index = commands.add_parser("index", help="build the FAISS index from the stored embeddings")
    index.add_argument("--check", action="store_true", help="only check that the index matches the database")
    index.add_argument("--sample", type=int, help="with --check, compare the text of the first N records only")

    ask = commands.add_parser("ask", help="ask a question")
    ask.add_argument("question")

    args = parser.parse_args(argv)
    try:
        if args.command == "load":
            return api.load(args.deposit_dir)
        if args.command == "embed":
            options = []
            if args.batch_size is not None:
                options += ["--batch-size", str(args.batch_size)]
            if args.force:
                options.append("--force")
            for source in args.source:
                options += ["--source", source]
            if args.stale_only:
                options.append("--stale-only")
            return api.embed_records(*options)
        if args.command == "index":
            options = ["--check"] if args.check else []
            if args.sample is not None:
                options += ["--sample", str(args.sample)]
            return api.build_index(*options)
        print(api.ask(args.question)["answer"])
        return 0
    except (CommandError, ImproperlyConfigured) as exc:
        print(f"{parser.prog}: error: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
