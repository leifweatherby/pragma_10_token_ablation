# Pragma

This repository contains code for chain-of-thought generation experiments. See
[project drive][drive] for more.

[drive]: https://drive.google.com/drive/folders/1NaTwKkS502v6vdxGs9DI80AUx_xY62IY?usp=drive_link

**File tree**

```
.
├── app                     Web viewer for run outputs
│   ├── frontend            Web viewer code
│   └── scripts             Scripts to prepare run data for the viewer
├── config                  TOML files for generation runs
├── data                    [Untracked] Data directory
├── docs                    Project documentation
├── notebooks               [Untracked] Jupyter notebook analyses
├── pixi.lock               Pixi lockfile
├── pyproject.toml          Python project metadata
├── README.md               README
├── ruff.toml               Configuration for ruff
├── scripts                 Misc. scripts
├── src                     Python module
│   └── pragma
│       ├── config.py       Configuration classes for generation
│       ├── mcqa.py         Dataclasses for QA pairs and model responses
│       ├── results.py      Dataclasses for ablation results
│       ├── eval            Judge module
│       ├── generation      Chain-of-thought generation code
│       └── tokenize        Tokenization utilities
└── tests                   Project tests
```


## Installation

Steps:

1. Clone the repository

   ```sh
   git clone git@github.com:digitaltheorylab/pragma.git
   cd pragma
   ```

2. If you don't have [pixi][pixi] installed, install it

3. Install the pixi environment

   ```sh
   pixi install
   ```

[pixi]: https://pixi.sh/latest/


## Data Management

The project uses [rclone](https://rclone.org/) to sync data with Google Drive.

**Initial setup**

1. Authenticate with Google Drive (one-time setup):

   ```sh
   pixi run data-auth
   ```

   This will open a browser window to authenticate with your Google account. The
   remote will be configured to access the [shared data folder][data-drive].

2. Download data from Google Drive:

   ```sh
   pixi run data-pull
   ```

   This will copy all files from the shared Google Drive folder to your local
   `data/` directory.

[data-drive]: https://drive.google.com/drive/folders/13wRlxC5j7lzkaAA-kA1zFSFeKSGMr73s?usp=drive_link

**Syncing data**

- To download new data from Google Drive: `pixi run data-pull`
- To upload your local results to Google Drive: `pixi run data-push`

The `data/` directory is gitignored and will not be tracked in version control.


## Experiments

Steps:

1. Create a configuration file in `config/` following the template in
   [configuration docs](docs/config.md)

2. (Optional) Ensure you have an API key stored at the environment variable
   specified in your configuration file

   ```sh
   export ANTHROPIC_API_KEY=<your-key>
   ```

3. Run CoT. Use `-r onepass` to do run a single pass of QA pairs with CoT-OFF
   and CoT-ON

   ```sh
   pixi r python -m pragma -r onepass config/<your-config.toml>
   ```
    
   Use `-r ablations` to kick off the ablations

   ```sh
   pixi r python -m pragma -r ablations config/<your-config.toml>
   ```

   Results for either run type will be stored under the subdirectory specified
   in your configuration file. See [run types docs](docs/runtypes.md) and
   [outputs docs](docs/outputs.md)


## Viewing Results

The `app/` directory contains a static web application for viewing and
analyzing run outputs.

**Quick Start**

After pulling data, run the combined setup task:

1. Pull data from Google Drive

   ```sh
   pixi r data-pull
   ```

2. Install frontend dependencies (if needed) and prepare data. **Note:** You
   will need `npm` installed

   ```sh
   pixi r app-setup
   ```

3. Start the viewer, then open http://localhost:5173 in your browser

   ```sh
   pixi r app-dev
   ```

**Note:** The `app-setup` task extracts the tokenizer vocabulary and prepares
all run data as static JSON files. This enables the viewer to display n-grams
as human-readable text (e.g., "the cat") instead of token IDs (e.g., `[198,
32313]`). Re-run `app-setup` whenever you add new runs or switch to a different
model.

For more information about the app, see the [documentation](docs/app.md).

## Contributing

**Branching**

+ Use git branches to develop new features or fixes
+ Branch names should be descriptive, e.g. `pmi-metric`
+ No need for a pull request unless you want a new set of eyes on something

**Code style**

+ Follow the [NumPy style guide][np] for all docstrings
+ Format code with `ruff`:

  ```sh
  pixi r ruff format <code.py>
  ```

[np]: https://numpydoc.readthedocs.io/en/latest/format.html


## Reasoning Complexity Study (2026-02-09)

Investigation into when chain-of-thought reasoning helps vs. hurts performance.

### Key Finding

**Reasoning benefits are task-complexity dependent:**

| Complexity | Effect | Delta Range |
|------------|--------|-------------|
| LOW | Thinking **hurts** | -31% to -51% |
| MEDIUM | Thinking **helps** | +1% to +15% |
| HIGH | Both struggle | -6% |

This validates findings from [Apple ML Research: "The Illusion of Thinking"](https://machinelearning.apple.com/research/illusion-of-thinking).

### Results

| Task | Complexity | CoT-OFF | CoT-ON | Delta |
|------|------------|---------|--------|-------|
| Word Unscrambling | LOW | 66% | 35% | -31% |
| Hyperbaton | LOW | 55% | 4% | -51% |
| ARC-Easy | MEDIUM | 46% | 47% | +1% |
| OpenBookQA | MEDIUM | 30% | 45% | +15% |
| Logical Deduction | HIGH | 18% | 12% | -6% |

### Critical Discovery: Reasoning is Fragile

On medium-complexity tasks, ablating just **10 tokens** causes complete collapse:

- ARC-Easy: 47% → 0%
- OpenBookQA: 45% → 0%

Critical tokens: `"Okay"`, `"So"`, `"I"`, `"think"`, `"But"`, `"?"`, `"First"`, `"maybe"`, `"might"`, newlines.

These same tokens **hurt** on low-complexity tasks but **enable** reasoning on medium-complexity tasks.

### Study Files

- `data/complexity_study_report.html` - Interactive HTML report
- `data/session_transcript_2026-02-09.txt` - Full session transcript
- `config/qwen3-0.6B-*.toml` - Experiment configurations
