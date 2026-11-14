# LAMP Genie

Genie is the question-answering agent of the LAMP materials database (https://lampdb.gnu.ac.kr). It answers
questions about the 249,075 material records in LAMP by calling tools that search the whole database and run a few
physics models (tolerance factor, alloy band gap, band offsets). Questions can be in English or Korean.

This repository contains the agent, its tools and prompts, and the commands that load the database and build the
search index.

## Requirements

- Python 3.10 or later
- PostgreSQL 16
- An OpenAI API key
- About 60 GB of memory for building the search index

## Setup

```bash
git clone https://github.com/wntjr3364/lamp-genie.git
cd lamp-genie
pip install .
```

Download `lamp_public_database.zip` (271 MB) from https://doi.org/10.5281/zenodo.24222439, then load it into an empty
database and build the index:

```bash
createdb lamp
export DATABASE_URL=postgresql://user:password@localhost:5432/lamp
export OPENAI_API_KEY=sk-...

unzip lamp_public_database.zip
python -m lamp_genie load lamp_public_database    # creates the tables and loads the CSV files
python -m lamp_genie embed                        # embeds every record with text-embedding-3-large
python -m lamp_genie index                        # writes the FAISS index to ./faiss_index
```

`embed` sends about 38 million tokens to the OpenAI embedding API and stores the vectors in the database, so it
only has to run once. `index` takes about 6 minutes. Set `LAMP_INDEX_DIR` to keep the index somewhere else.

## Usage

```bash
python -m lamp_genie ask "Which lead-free halide perovskites have a band gap between 1.2 and 1.5 eV?"
```

```python
import lamp_genie

result = lamp_genie.ask("What is the band gap of CsSnI3 in the database, by source?")
print(result["answer"])
```

`result["tool_results"]` holds the raw tool outputs. For recommendation and similarity questions,
`result["materials"]` holds the matching records.

The chat model is `gpt-4o-mini` (`OPENAI_CHAT_MODEL`). `python -m lamp_genie <command> --help` lists the remaining
options.

## Tools

The agent chooses among 13 tools, defined in `materials/services/rag.py`:

`recommend_materials`, `similarity_search`, `material_comparison`, `lookup_property`, `database_inventory`,
`materials_retriever`, `screen_candidates`, `predict_bandgap`, `analyze_stability`, `calculate_band_alignment`,
`calculate_properties`, `validate_formula`, `visualize_structure`

## Data

LAMP combines Materials Project, OQMD, JARVIS-DFT, JARVIS superconductors, 2DMatPedia, C2DB, QMOF, the Harvard
Clean Energy Project, HOPV15, the Perovskite Database, Starrydata2, and two sets of values curated from the
literature (perovskites and OLED emitters). Each source keeps its own license. Please cite the sources you use.

## License

MIT
