# generated_stage1__01_the_machine_learning_landscape__20260417_193619

- run_id: `generated_stage1__01_the_machine_learning_landscape__20260417_193619`
- prompt_file: `codex_prompts/generated_stage1__01_the_machine_learning_landscape.md`
- prompt_stem: `generated_stage1__01_the_machine_learning_landscape`
- started_at_utc: `20260417_193619`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260417_193725`
- runner: `tools/codex/run_prompt.py`
- return_code: `0`
- retry_of_run_id:

## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `66.432`
- final_output_char_count: `478`
- stderr_char_count: `215444`

## Prompt Text

```md
# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: 01_the_machine_learning_landscape.ipynb
- Chapter: 1
- Notebook stem: 01_the_machine_learning_landscape

## Current Chapter Intro State
Status: THIN at cell index 0 (~71 words).

Current content (if any):
````

**Chapter 1 – The Machine Learning landscape**

_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._...

```

## Your Task
REPLACE the markdown cell at index 0 with the full chapter intro.

The new intro must follow the treatment specification below exactly.

## Treatment Specification

#### Treatment structure for the entire notebook:

```
[CHAPTER INTRO MARKDOWN CELL]
  - What is this chapter about?
  - What are the 3-5 main concepts a student should walk away understanding?
  - Why does this topic matter in the broader ML landscape?
  - Where does it sit relative to what came before and what comes next?
  - Any key vocabulary terms to know before diving in
  (Aim for 300-500 words. This is the "sit down, let me tell you what we're about to do" cell.)

Then for each logical section or code cell in the notebook:

[GOAL MARKDOWN CELL — before the code]
  - What is the goal of the next code block?
  - Why does this matter for ML? (Not just "it runs the model" — why do we care?)
  - What is this code doing that is a *better practice* worth noting?
  - What is this code doing that is just *plumbing* (necessary but not pedagogically deep)?
  (Aim for 4-8 sentences. Make the distinction between "this is important" and "this is boilerplate" explicit.)

[PYTHON CODE CELL]
  (unchanged)

[IMPLEMENTATION DETAIL MARKDOWN CELL — after the code]
  - What did we just see happen?
  - What are the implementation choices worth noticing? (e.g., why this hyperparameter,
    why this data split ratio, why this particular sklearn API call?)
  - What might go wrong here in practice, and how would you know?
  - If there is output, what should the student be looking for in that output?
  (Aim for 3-6 sentences. This is the "here's what's interesting about what we just ran" cell.)
```

## Gold Standard Example (from 06_decision_trees.ipynb — the finished notebook)

**Chapter Intro Example:**
## Chapter Overview: Decision Trees

Decision trees are one of the most intuitive and interpretable machine learning algorithms you will encounter. They make predictions by asking a sequence of yes/no questions about the input features, following branches down to a leaf node that contains the final answer. If you have ever played Twenty Questions, you already understand the core idea.

**What you should walk away understanding:**

1. **How decision trees make decisions** — the splitting criterion (Gini impurity and entropy), how the tree chooses which feature to split on at each node, and how it decides when to stop splitting.
2. **Regularization and the overfitting problem** — unconstrained trees will memorize training data perfectly and generalize poorly. Hyperparameters like `max_depth`...

**Goal/Code/Implementation Trio Example:**
### GOAL BEFORE CELL:
### Goal Before This Cell

**Goal:** Verify the Python environment and notebook prerequisites before doing any modeling.

**Why this matters for machine learning:** This cell contributes to the larger workflow of building, inspecting, evaluating, or explaining a decision tree model.

### CODE CELL:
```python
import sys

assert sys.version_info >= (3, 7)

#%pip install graphviz
#Note: Using %pip (with the percent sign) is better than !pip inside notebooks because it guarantees installation into the kernel's specific virtual environment rather than a global system Python.
```

### IMPLEMENTATION NOTES AFTER CELL:
### Implementation Notes After This Cell

Failing fast on environment problems is a practical ML skill because many notebook errors come from setup drift rather than model logic.

**Broader skill:** Being able to connect model behavior to code-level implementation details is one of the most valuable habits you can build in machine learning.

## Output Contract

1. Read the full notebook from: `01_the_machine_learning_landscape.ipynb`
2. REPLACE the markdown cell at index 0 with the full chapter intro.
3. Write the COMPLETE modified notebook as valid JSON back to: `01_the_machine_learning_landscape.ipynb`
4. Do NOT modify any `code` cells — not their source, outputs, or metadata
5. Do NOT add, remove, or reorder any cell other than the one change described above
6. The new intro cell must have `"cell_type": "markdown"` and a `"source"` field
7. Preserve valid Jupyter notebook structure and preserve all untouched cells exactly as they are

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- The intro must be 300–500 words covering: what this chapter is about, 3–5 learning
  objectives, why this topic matters in ML, where it fits in the course, key vocabulary
- Write clean JSON — the same structure as the input notebook
- The output file must be a valid Jupyter notebook that can be opened in Jupyter Lab
```

## Codex Final Output

Replaced cell 0 in [01_the_machine_learning_landscape.ipynb](/home/jevert/git/handson-ml3/01_the_machine_learning_landscape.ipynb) with a 403-word chapter intro that covers the chapter purpose, five learning objectives, why the topic matters in the broader ML landscape, where it fits in the course, and key vocabulary.

Validation passed: the notebook still parses as valid JSON, cell 0 remains a `markdown` cell, the notebook still has 53 cells, and no code cells were edited.

## Stderr

```text
Runner context:
- sandbox: workspace-write
- full_auto: False
- model: (default)
- profile: (default)
- json: False
- codex_command: codex exec -C /home/jevert/git/handson-ml3 --sandbox workspace-write --output-last-message /tmp/codex-last-message-l_v_0nv3.txt -
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /home/jevert/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d9cf1-b5ae-7e03-ad45-f7a4f84ebf0a
--------
user
# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: 01_the_machine_learning_landscape.ipynb
- Chapter: 1
- Notebook stem: 01_the_machine_learning_landscape

## Current Chapter Intro State
Status: THIN at cell index 0 (~71 words).

Current content (if any):
````

**Chapter 1 – The Machine Learning landscape**

_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._...

```

## Your Task
REPLACE the markdown cell at index 0 with the full chapter intro.

The new intro must follow the treatment specification below exactly.

## Treatment Specification

#### Treatment structure for the entire notebook:

```
[CHAPTER INTRO MARKDOWN CELL]
  - What is this chapter about?
  - What are the 3-5 main concepts a student should walk away understanding?
  - Why does this topic matter in the broader ML landscape?
  - Where does it sit relative to what came before and what comes next?
  - Any key vocabulary terms to know before diving in
  (Aim for 300-500 words. This is the "sit down, let me tell you what we're about to do" cell.)

Then for each logical section or code cell in the notebook:

[GOAL MARKDOWN CELL — before the code]
  - What is the goal of the next code block?
  - Why does this matter for ML? (Not just "it runs the model" — why do we care?)
  - What is this code doing that is a *better practice* worth noting?
  - What is this code doing that is just *plumbing* (necessary but not pedagogically deep)?
  (Aim for 4-8 sentences. Make the distinction between "this is important" and "this is boilerplate" explicit.)

[PYTHON CODE CELL]
  (unchanged)

[IMPLEMENTATION DETAIL MARKDOWN CELL — after the code]
  - What did we just see happen?
  - What are the implementation choices worth noticing? (e.g., why this hyperparameter,
    why this data split ratio, why this particular sklearn API call?)
  - What might go wrong here in practice, and how would you know?
  - If there is output, what should the student be looking for in that output?
  (Aim for 3-6 sentences. This is the "here's what's interesting about what we just ran" cell.)
```

## Gold Standard Example (from 06_decision_trees.ipynb — the finished notebook)

**Chapter Intro Example:**
## Chapter Overview: Decision Trees

Decision trees are one of the most intuitive and interpretable machine learning algorithms you will encounter. They make predictions by asking a sequence of yes/no questions about the input features, following branches down to a leaf node that contains the final answer. If you have ever played Twenty Questions, you already understand the core idea.

**What you should walk away understanding:**

1. **How decision trees make decisions** — the splitting criterion (Gini impurity and entropy), how the tree chooses which feature to split on at each node, and how it decides when to stop splitting.
2. **Regularization and the overfitting problem** — unconstrained trees will memorize training data perfectly and generalize poorly. Hyperparameters like `max_depth`...

**Goal/Code/Implementation Trio Example:**
### GOAL BEFORE CELL:
### Goal Before This Cell

**Goal:** Verify the Python environment and notebook prerequisites before doing any modeling.

**Why this matters for machine learning:** This cell contributes to the larger workflow of building, inspecting, evaluating, or explaining a decision tree model.

### CODE CELL:
```python
import sys

assert sys.version_info >= (3, 7)

#%pip install graphviz
#Note: Using %pip (with the percent sign) is better than !pip inside notebooks because it guarantees installation into the kernel's specific virtual environment rather than a global system Python.
```

### IMPLEMENTATION NOTES AFTER CELL:
### Implementation Notes After This Cell

Failing fast on environment problems is a practical ML skill because many notebook errors come from setup drift rather than model logic.

**Broader skill:** Being able to connect model behavior to code-level implementation details is one of the most valuable habits you can build in machine learning.

## Output Contract

1. Read the full notebook from: `01_the_machine_learning_landscape.ipynb`
2. REPLACE the markdown cell at index 0 with the full chapter intro.
3. Write the COMPLETE modified notebook as valid JSON back to: `01_the_machine_learning_landscape.ipynb`
4. Do NOT modify any `code` cells — not their source, outputs, or metadata
5. Do NOT add, remove, or reorder any cell other than the one change described above
6. The new intro cell must have `"cell_type": "markdown"` and a `"source"` field
7. Preserve valid Jupyter notebook structure and preserve all untouched cells exactly as they are

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- The intro must be 300–500 words covering: what this chapter is about, 3–5 learning
  objectives, why this topic matters in ML, where it fits in the course, key vocabulary
- Write clean JSON — the same structure as the input notebook
- The output file must be a valid Jupyter notebook that can be opened in Jupyter Lab

codex
I’m updating the chapter intro in `01_the_machine_learning_landscape.ipynb` and leaving every code cell untouched. First I’m reading the notebook structure and the existing cell 0 JSON so I can replace only that markdown cell cleanly.
exec
/bin/bash -lc "rg -n 'Chapter 1|The Machine Learning landscape|cells' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
2:  "cells": [
9:        "**Chapter 1 – The Machine Learning landscape**\n",

exec
/bin/bash -lc "sed -n '1,120p' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Zs3Na-QS4i2S"
      },
      "source": [
        "**Chapter 1 – The Machine Learning landscape**\n",
        "\n",
        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
        "\n",
        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gZ1qzcpg4i2T"
      },
      "source": [
        "<table align=\"left\">\n",
        "  <td>\n",
        "    <a href=\"https://colab.research.google.com/github/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
        "  </td>\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb\"><img src=\"https://kaggle.com/static/images/open-in-kaggle.svg\" /></a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hlGz6n-d4i2T"
      },
      "source": [
        "# Setup"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iOiz2Xcj4i2T"
      },
      "source": [
        "This project requires Python 3.7 or above:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sjlzbAwV4i2T"
      },
      "outputs": [],
      "source": [
        "import sys\n",
        "\n",
        "assert sys.version_info >= (3, 7)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zSAodgEZ4i2U"
      },
      "source": [
        "Scikit-Learn ≥1.0.1 is required:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Oov_u0z24i2U"
      },
      "outputs": [],
      "source": [
        "from packaging import version\n",
        "import sklearn\n",
        "\n",
        "assert version.parse(sklearn.__version__) >= version.parse(\"1.0.1\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hANbb6MR4i2U"
      },
      "source": [
        "Let's define the default font sizes, to plot pretty figures:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gGlNWVI24i2U"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.rc('font', size=12)\n",
        "plt.rc('axes', labelsize=14, titlesize=14)\n",
        "plt.rc('legend', fontsize=12)\n",
        "plt.rc('xtick', labelsize=10)\n",
        "plt.rc('ytick', labelsize=10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SpHM5Gds4i2U"
      },
      "source": [
        "Make this notebook's output stable across runs:"
      ]
    },

codex
I have the exact JSON shape for cell 0. I’m scanning the notebook headings now so the new intro matches what this chapter actually covers rather than writing a generic ML overview.
exec
/bin/bash -lc "rg -n '"'^'"\\s*\"#|"'^'"\\s*\"##|"'^'"\\s*\"###' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
38:        "# Setup"
140:        "# Code example 1-1"
151:        "# -------------------------\n",
152:        "# 1) Load life satisfaction dataset (Ageron)\n",
153:        "# -------------------------\n",
157:        "# Use 1D arrays for convenience\n",
162:        "# -------------------------\n",
163:        "# 2) Fit baseline model and compute residuals\n",
164:        "# -------------------------\n",
175:        "# -------------------------\n",
176:        "# 3) List countries above / below the best-fit line\n",
177:        "# -------------------------\n",
190:        "# -------------------------\n",
191:        "# 4) Pull population data (OWID) and merge it in\n",
192:        "#    OWID CO2 dataset includes: country, year, iso_code, population ...\n",
193:        "# -------------------------\n",
197:        "# Remove aggregate regions (OWID_* codes)\n",
208:        "# Country-name fixes (add to this list if you see missing populations)\n",
220:        "# Report any countries that failed to match population\n",
226:        "# Keep only rows with population so terciles work\n",
229:        "# -------------------------\n",
230:        "# 5) Split into population terciles and fit separate lines\n",
231:        "# -------------------------\n",
276:        "# -------------------------\n",
277:        "# 6) BONUS OPTION: population-weighted best-fit line (each person counts, not each country)\n",
278:        "# -------------------------\n",
406:        "# -------------------------\n",
407:        "# 1) Load data\n",
408:        "# -------------------------\n",
415:        "# For sklearn, features must be 2D\n",
418:        "# -------------------------\n",
419:        "# 2) Fit initial model\n",
420:        "# -------------------------\n",
427:        "# Standardized residuals (how many \"sigmas\" away from the line)\n",
431:        "# -------------------------\n",
432:        "# 3) Influence stats (Cook's distance) for simple linear regression\n",
433:        "#    This is a common statistical way to flag points that overly influence the line.\n",
434:        "# -------------------------\n",
438:        "# Design matrix with intercept: [x, 1]\n",
447:        "# Thresholds (tweakable)\n",
453:        "# Add diagnostics to a new DataFrame for easy viewing\n",
462:        "# Print the most suspicious points\n",
474:        "# -------------------------\n",
475:        "# 4) Plot: highlight outliers and show best-fit line(s)\n",
476:        "# -------------------------\n",
487:        "# Inliers\n",
494:        "# Outliers\n",
501:        "# Best-fit line (all points)\n",
504:        "# Optional: refit without outliers and show the \"cleaned\" line\n",
511:        "# Label outlier countries on the plot (if the column exists)\n",
522:        "# Add equation box\n",
536:        "# -------------------------\n",
537:        "# 5) Extra diagnostic plot: Cook's distance\n",
538:        "# -------------------------\n",
599:        "# Download and prepare the data\n",
603:        "# Feature (input): GDP per capita\n",
606:        "# Target (output): reported life satisfaction score\n",
609:        "# Select and train a linear model\n",
613:        "# Define plot range for the line of best fit\n",
616:        "# Generate GDP values across the plot range and predict corresponding satisfaction values\n",
620:        "# Extract equation coefficients for: y = mx + b\n",
625:        "# Visualize the data\n",
633:        "# Plot the best-fit line\n",
641:        "# Set plot bounds\n",
645:        "# Add equation text to the plot\n",
657:        "# Make a prediction for Cyprus\n",
724:        "# Download and prepare the data\n",
730:        "# Visualize the data\n",
736:        "# Select a linear model\n",
739:        "# Train the model\n",
742:        "# Make a prediction for Cyprus\n",
788:        "# Select a 3-Nearest Neighbors regression model\n",
793:        "# Train the model\n",
796:        "# Make a prediction for Cyprus\n",
806:        "# Generating the data and figures — please skip"
837:        "# Where to save the figures\n",
854:        "## Load and prepare Life satisfaction data"
2326:        "# Exercise Solutions"

exec
/bin/bash -lc "sed -n '120,260p' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jYWs7wFF4i2U"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "np.random.seed(42)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vcvdkOIz4i2U"
      },
      "source": [
        "# Code example 1-1"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# -------------------------\n",
        "# 1) Load life satisfaction dataset (Ageron)\n",
        "# -------------------------\n",
        "data_root_url = \"https://github.com/ageron/data/raw/main/\"\n",
        "life_satisfaction_df = pd.read_csv(data_root_url + \"lifesat/lifesat.csv\")\n",
        "\n",
        "# Use 1D arrays for convenience\n",
        "gdp_per_capita_usd = life_satisfaction_df[\"GDP per capita (USD)\"].to_numpy()\n",
        "life_satisfaction_score = life_satisfaction_df[\"Life satisfaction\"].to_numpy()\n",
        "gdp_per_capita_usd_2d = gdp_per_capita_usd.reshape(-1, 1)\n",
        "\n",
        "# -------------------------\n",
        "# 2) Fit baseline model and compute residuals\n",
        "# -------------------------\n",
        "baseline_model = LinearRegression()\n",
        "baseline_model.fit(gdp_per_capita_usd_2d, life_satisfaction_score)\n",
        "\n",
        "baseline_predicted = baseline_model.predict(gdp_per_capita_usd_2d)\n",
        "residuals = life_satisfaction_score - baseline_predicted  # + means above line, - means below line\n",
        "\n",
        "diagnostics_df = life_satisfaction_df.copy()\n",
        "diagnostics_df[\"Predicted satisfaction\"] = baseline_predicted\n",
        "diagnostics_df[\"Residual (actual - predicted)\"] = residuals\n",
        "\n",
        "# -------------------------\n",
        "# 3) List countries above / below the best-fit line\n",
        "# -------------------------\n",
        "above_line_df = diagnostics_df[diagnostics_df[\"Residual (actual - predicted)\"] > 0].copy()\n",
        "below_line_df = diagnostics_df[diagnostics_df[\"Residual (actual - predicted)\"] < 0].copy()\n",
        "\n",
        "above_line_df = above_line_df.sort_values(\"Residual (actual - predicted)\", ascending=False)\n",
        "below_line_df = below_line_df.sort_values(\"Residual (actual - predicted)\")  # most negative first\n",
        "\n",
        "print(\"\\n=== Countries ABOVE the best-fit line (happier than predicted) ===\")\n",
        "print(above_line_df[[\"Country\", \"GDP per capita (USD)\", \"Life satisfaction\", \"Residual (actual - predicted)\"]].to_string(index=False))\n",
        "\n",
        "print(\"\\n=== Countries BELOW the best-fit line (less happy than predicted) ===\")\n",
        "print(below_line_df[[\"Country\", \"GDP per capita (USD)\", \"Life satisfaction\", \"Residual (actual - predicted)\"]].to_string(index=False))\n",
        "\n",
        "# -------------------------\n",
        "# 4) Pull population data (OWID) and merge it in\n",
        "#    OWID CO2 dataset includes: country, year, iso_code, population ...\n",
        "# -------------------------\n",
        "owid_co2_url = \"https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv\"\n",
        "owid_df = pd.read_csv(owid_co2_url, usecols=[\"country\", \"year\", \"iso_code\", \"population\"])\n",
        "\n",
        "# Remove aggregate regions (OWID_* codes)\n",
        "owid_df = owid_df[~owid_df[\"iso_code\"].astype(str).str.startswith(\"OWID\")].copy()\n",
        "\n",
        "population_year = 2020  # choose a year to line up with your GDP example year\n",
        "population_df = (\n",
        "    owid_df[owid_df[\"year\"] == population_year][[\"country\", \"population\"]]\n",
        "    .dropna()\n",
        "    .drop_duplicates(subset=[\"country\"])\n",
        "    .rename(columns={\"country\": \"JoinCountry\", \"population\": \"Population\"})\n",
        ")\n",
        "\n",
        "# Country-name fixes (add to this list if you see missing populations)\n",
        "country_name_fixes = {\n",
        "    \"Korea\": \"South Korea\",\n",
        "    \"Czech Republic\": \"Czechia\",\n",
        "    \"Slovak Republic\": \"Slovakia\",\n",
        "    \"Russian Federation\": \"Russia\",\n",
        "}\n",
        "\n",
        "life_satisfaction_df[\"JoinCountry\"] = life_satisfaction_df[\"Country\"].replace(country_name_fixes)\n",
        "\n",
        "merged_df = life_satisfaction_df.merge(population_df, on=\"JoinCountry\", how=\"left\")\n",
        "\n",
        "# Report any countries that failed to match population\n",
        "missing_pop = merged_df[merged_df[\"Population\"].isna()][\"Country\"].tolist()\n",
        "if missing_pop:\n",
        "    print(\"\\n=== NOTE: These countries are missing population matches (add to country_name_fixes) ===\")\n",
        "    print(missing_pop)\n",
        "\n",
        "# Keep only rows with population so terciles work\n",
        "merged_df = merged_df.dropna(subset=[\"Population\"]).copy()\n",
        "\n",
        "# -------------------------\n",
        "# 5) Split into population terciles and fit separate lines\n",
        "# -------------------------\n",
        "merged_df[\"Population group\"] = pd.qcut(\n",
        "    merged_df[\"Population\"],\n",
        "    3,\n",
        "    labels=[\"Smallest 1/3 (population)\", \"Middle 1/3 (population)\", \"Largest 1/3 (population)\"]\n",
        ")\n",
        "\n",
        "plot_gdp_min_usd, plot_gdp_max_usd = 23_500, 62_500\n",
        "gdp_for_line = np.linspace(plot_gdp_min_usd, plot_gdp_max_usd, 200).reshape(-1, 1)\n",
        "\n",
        "for pop_group, group_df in merged_df.groupby(\"Population group\", sort=False):\n",
        "    group_gdp_2d = group_df[[\"GDP per capita (USD)\"]].to_numpy()\n",
        "    group_satisfaction = group_df[\"Life satisfaction\"].to_numpy()\n",
        "\n",
        "    group_model = LinearRegression()\n",
        "    group_model.fit(group_gdp_2d, group_satisfaction)\n",
        "\n",
        "    group_line = group_model.predict(gdp_for_line)\n",
        "\n",
        "    slope = group_model.coef_[0]\n",
        "    intercept = group_model.intercept_\n",
        "    r2 = group_model.score(group_gdp_2d, group_satisfaction)\n",
        "\n",
        "    equation_text = f\"Life satisfaction = {slope:.2e}*(GDP) + {intercept:.2f}   (R²={r2:.3f}, n={len(group_df)})\"\n",
        "\n",
        "    fig, ax = plt.subplots()\n",
        "    ax.scatter(group_df[\"GDP per capita (USD)\"], group_df[\"Life satisfaction\"])\n",
        "    ax.plot(gdp_for_line, group_line, linewidth=2)\n",
        "\n",
        "    # label points (handy when n is small like this dataset)\n",

exec
/bin/bash -lc "sed -n '560,830p' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "=== Potential outliers / influential points (ranked) ===\n",
            "      Country  GDP per capita (USD)  Life satisfaction  Residual  Std residual  Cook's distance\n",
            "United States          60235.728492                6.9 -0.932369     -2.331892         0.708694\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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VZcuXZS9vb167rnnTPYxr+99UyNV8+bNUx06dFDu7u7Kzs5OBQQEqLVr1xodNykpST377LOqatWqysnJSQ0aNEhdunTJaES0oN8xx44dU4DasGGDyX6IsiUjVSa8/vrrNGrUiCVLlvDWW2/h5+dHvXr1APjjjz/o168fbdq0YebMmVhYWLBs2TIeeOABdu/eTVBQUK7HXbp0KdOmTWPYsGE899xzJCcnc/z4cf7++28effRRQDdZfN++fQQHB1OnTh0iIyNZvHgx3bt3Jzw8HAcHB7p27cq0adP45JNPeO2112jSpAmA/r+FPe8zzzxD7dq1mTt3LtOmTSMwMJCaNWsCsG3bNi5cuMC4cePw8PDgn3/+YcmSJfzzzz8cOHBAP4H0ypUrBAUFcefOHSZMmEDjxo25fPky69atIykpqdBtvnfvHt27d+fcuXNMnToVPz8/1q5dy9ixY7lz5w7PPfecwfY//PADd+/e5ZlnnkGj0fDBBx8wdOhQLly4kOtoRNu2balbty5r1qxhzJgxButWr15NlSpV6NOnDwATJ05k3bp1TJ06FX9/f27evMmePXs4deoUAQEBub7mv/zyC/Xr16ddu3a5blNYo0ePZsmSJfz+++/07t07z23Pnz8PQNWqVfM97tSpU3Fzc2PWrFmcPn2axYsXc/HiRXbt2qV/nVesWMGYMWPo06cP77//PklJSSxevJjOnTtz5MgRg8m86enp9OnTh86dO/Phhx/mOjoA+b9/bGxsuHDhAhs3bmT48OH4+flx7do1vvzyS7p160Z4eLjRSNycOXPQaDS88sorxMbGsnDhQnr16sXRo0ext7c3akP16tVZvHgxkyZNYsiQIQwdOhSAFi1aAAX/LJhy+fJloqKijN4rXbp04aOPPuKff/6hWbNmAOzevRsLCwt2797NtGnT9MsAunbtavL4zzzzDFeuXGHbtm2sWLHC5DZF+YwAXLt2jY4dO5KUlMS0adOoWrUq3377LQ899BDr1q1jyJAhue5ryuuvv05cXByXLl3io48+Aij0iMucOXN44403GDFiBE899RTXr1/n008/pWvXrhw5csRgpOnmzZv069eP4OBgHn/8cf13m6l25fa9b8rHH3/MQw89xGOPPUZqaiohISEMHz6cX375hQEDBui3Gzt2LGvWrGH06NG0b9+eP//802B9poJ+x/j7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7dv6Y6xbt075+/srGxsb5ePjoz788EODc/j4+Kg5c+aocePGKScnJ+Xl5aW+/PLLXJ/P4vDKK6+ozp07G7z2hfXjjz8qjUajIiMj9csGDBigxo8fb7Dd0KFD1WOPPaZ/nJycrF588UVVq1Yt5eDgoIKCgtTOnTvzPNfLL7+smjZtarBs5MiRqk+fPvrHM2fOVC1btsz1GHv37lWdOnVSr7zyiurZs6fq3bu3euGFF/Sv+ZQpU9SwYcMM9nn33XeVhYWFOnz4sMljarVa/f+b+mwppdSOHTsUoJYuXapf5uPjoz766CP947S0NOXg4KBeffVVk+fJfF/98ssvqnnz5srW1la1a9dOnThxwmC73bt3q86dOys7OztVp04d9eyzz6qEhASD87711ltq9OjRytnZWY0aNcrka5+RkaHef/99Va9ePWVjY6O8vLzUO++8o1+f12dWqazX4osvvlB16tRR9vb2avjw4erOnTsmn6/cPk/p6elq/PjxytfXV9nZ2amGDRuqhQsXGrX322+/VXXq1DH53OXGx8dHbdq0qUjvfaWyvoO3b99e4H2ee+45Va9ePYP3Tbdu3dRzzz1XoP2fffZZNX369MI21UhqaqrauHGjwWtmTsq6//n9fgulKs1IVXYxMTE0bdqUF198kZiYGF566SXS0tLo06cPzs7O7N69m7179+Lk5ETfvn1z/VfvG2+8QXh4OFu2bOHUqVMsXryYatWq6dc7OzuzfPlywsPD+fjjj1m6dCkfffQRACNHjuTFF180GIEYOXKk0Tk++eQTfv75Z9asWcPp06dZuXIlvr6+gO5fi3379mXEiBHExMTw8ccfA7ryDwsXLmT//v0sW7aMP/74g5dffll/zKNHj9KzZ0/8/f3Zv38/e/bsYdCgQWRkZPDxxx/ToUMHg39henl5GbXr0KFDjBgxguDgYE6cOMGsWbN44403jC5dzZ8/n7Zt23LkyBEmT57MpEmTOH36dK6vzdy5c3FycsrzLyoqKtf9f/75Z9q2bcuIESNo0KABbdq0YenSpblub8rXX39Nr1698PHx0S/r2LEjO3bs0F9iPXbsGHv27DG4TDt16lT2799PSEgIx48fZ/jw4fTt25ezZ8/meq79+/fTq1cvg2V9+vRh//79BsvOnj1LrVq1qFu3Lo899pjBc9CxY0d27tzJ/v372bVrF889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822jVqtl+fLl+nlXo0ePZseOHcyZMyfffffs2UNoaCixsbHY2toC8OGHH7Jx40bWrVunn3uVmprKd999p790ePjwYdLT0xk6dCg+Pj4ANG/e3ODYtWrVAuDixYv4+vrm2xYhKgIZqRJClHuxd3MPqIqyXXFRShkkS9y+fTs9e/akdu3aODs7M3r0aG7evElSUlKexzl06BCDBg3C29sbZ2dnunXrBugCp9zcu3dPf6ktL76+vvqACnQ13fK7tJjp2LFjJCQkULVqVZycnPR/ERERnD9/Xr+dj4+PwVysli1b0rNnT5o3b87w4cNZunQpt2/fNjh25tyv/J4bISoSCaqEEOVeDef8g4fCbFdcTp06hZ+fHwCRkZEMHDiQFi1asH79eg4dOsTnn38O6EZycpOYmEifPn1wcXFh5cqVhIWF8eOPP+a7X7Vq1YwCFVOsra0NHms0mgLftZiQkICnpydHjx41+Dt9+jTTp0/Xb+fo6Giwn6WlJdu2bWPLli34+/vz6aef0qhRIyIiIvTbZF4+vN+J8UKUJ3L5TwhR7gX5uePpasfVuGST86o0gIerHUF+7qXWpj/++IMTJ07wwgsvALrRJq1Wy/z58/XFbtesWWOwj42NDRkZGQbL/v33X27evMl7772Hl5cXAAcPHsz3/K1atSI8PLw4upKrgIAArl69ipWVVaEv0Wk0Gjp16kSnTp1488038fHx4ccff+R///sfACdPnsTa2pqmTaXeq6g8ZKRKCFHuWVpomDnIH9AFUNllPp45yB9Li5KpW5aSksLVq1e5fPkyhw8fZu7cuTz88MMMHDiQJ554AoD69euTlpbGp59+yoULF1ixYgVffPGFwXF8fX1JSEhgx44d3Lhxg6SkJLy9vbGxsdHv9/PPP/P222/n26YHH3yQ/fv3GwVpxalXr1506NCBwYMH8/vvvxMZGcm+fft4/fXX8wz8/v77b+bOncvBgweJiopiw4YNXL9+nSZNmui32b17N126dLnvFBBClCcSVAkhKoS+zTxZ/HgANV0ML/F5uNqx+PGAEs1TtXXrVjw9PfH19aVv377s3LmTTz75hJ9++glLS0tAN49owYIFvP/++zRr1oyVK1fy7rvvGhynY8eOTJw4kZEjR1K9enU++OADqlevzvLly1m7di3+/v689957fPjhh/m2qV+/flhZWbF9+/YS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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# Download and prepare the data\n",
        "data_root_url = \"https://github.com/ageron/data/raw/main/\"\n",
        "life_satisfaction_df = pd.read_csv(data_root_url + \"lifesat/lifesat.csv\")\n",
        "\n",
        "# Feature (input): GDP per capita\n",
        "gdp_per_capita_usd = life_satisfaction_df[[\"GDP per capita (USD)\"]].values\n",
        "\n",
        "# Target (output): reported life satisfaction score\n",
        "life_satisfaction_score = life_satisfaction_df[[\"Life satisfaction\"]].values\n",
        "\n",
        "# Select and train a linear model\n",
        "life_satisfaction_model = LinearRegression()\n",
        "life_satisfaction_model.fit(gdp_per_capita_usd, life_satisfaction_score)\n",
        "\n",
        "# Define plot range for the line of best fit\n",
        "gdp_min_usd, gdp_max_usd = 23_500, 62_500\n",
        "\n",
        "# Generate GDP values across the plot range and predict corresponding satisfaction values\n",
        "gdp_values_for_best_fit_line = np.linspace(gdp_min_usd, gdp_max_usd, 200).reshape(-1, 1)\n",
        "predicted_satisfaction_for_best_fit_line = life_satisfaction_model.predict(gdp_values_for_best_fit_line)\n",
        "\n",
        "# Extract equation coefficients for: y = mx + b\n",
        "slope_m = life_satisfaction_model.coef_[0][0]\n",
        "intercept_b = life_satisfaction_model.intercept_[0]\n",
        "best_fit_equation_text = f\"y = {slope_m:.2e}x + {intercept_b:.2f}\"\n",
        "\n",
        "# Visualize the data\n",
        "ax = life_satisfaction_df.plot(\n",
        "    kind=\"scatter\",\n",
        "    grid=True,\n",
        "    x=\"GDP per capita (USD)\",\n",
        "    y=\"Life satisfaction\"\n",
        ")\n",
        "\n",
        "# Plot the best-fit line\n",
        "ax.plot(\n",
        "    gdp_values_for_best_fit_line,\n",
        "    predicted_satisfaction_for_best_fit_line,\n",
        "    linewidth=2,\n",
        "    label=\"Line of best fit\"\n",
        ")\n",
        "\n",
        "# Set plot bounds\n",
        "ax.set_xlim(gdp_min_usd, gdp_max_usd)\n",
        "ax.set_ylim(4, 9)\n",
        "\n",
        "# Add equation text to the plot\n",
        "ax.text(\n",
        "    gdp_min_usd + 1000,\n",
        "    8.7,\n",
        "    best_fit_equation_text,\n",
        "    fontsize=11,\n",
        "    bbox=dict(boxstyle=\"round,pad=0.3\", facecolor=\"white\", alpha=0.8)\n",
        ")\n",
        "\n",
        "ax.legend()\n",
        "plt.show()\n",
        "\n",
        "# Make a prediction for Cyprus\n",
        "cyprus_gdp_per_capita_2020_usd = [[37_655.2]]\n",
        "print(life_satisfaction_model.predict(cyprus_gdp_per_capita_2020_usd))\n"
      ],
      "metadata": {
        "id": "fPUpo7BR6axf",
        "outputId": "dc65c021-ce86-406c-8765-b9591ecae890",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        }
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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ODo6WsSVVZVKhaqqKty4cUNrZKi5M2buNVeJ8/Ly4O7u3ug/AC2mUCEiotrd3pzPUggh1NPVW0JhdrumyN3d3d0g3ZVZqBAREWQyGfz9/eHj46Pz9rg5UigU+P3339G7d2+Lu5pq7NxtbW0NdiudhQoREalZW1tbzHNq1tbWqK6uhr29vcUVKs0pd8u6KUdERETNCgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZJi1UlEolZs+ejVatWsHBwQGtW7fGm2++CSGEKcMiIiIiibAx5cHffvttrFixAl999RXat2+PQ4cOYeLEiXBzc8PUqVNNGRoRERFJgEkLlT///BPDhg3DkCFDAAChoaFYt24dkpOTTRkWERERSYRJC5UePXrgs88+w5kzZ9CmTRukpaVh7969ePfdd3VuX1lZicrKSvX3paWlAACFQgGFQtGoWGpe39j9NFeWnL8l5w4wf0vO35JzByw7f1PnfjanuMHbyoQJHwhRqVR444038M4778Da2hpKpRKLFi3C66+/rnP7efPmYf78+VrL165dC0dHR2OHS0RERHepSgmkFcqw/4oVzuZfR2bCKJSUlMDV1bXO15m0UElMTMTMmTOxdOlStG/fHqmpqXjppZfw7rvvYsKECVrb67qiEhQUhPz8/HoTrY9CoUBSUhIGDBgAW1vbRu2rObLk/C05d4D5W3L+lpw7YNn5N2Xup3LLsOFQFn5Ky0HpjWoAgKqyosGFiklv/cycOROvvfYa4uPjAQAdO3bEpUuXsHjxYp2Filwuh1wu11pua2trsDfakPtqjiw5f0vOHWD+lpy/JecOWHb+xsq9vLIam9MuIzElE2mZxVrrw7ydkNnAfZm0UKmoqICVleYIaWtra6hUKhNFRERERHdDCIG0rBIkJmdg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FgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJYqFCREREksVChYiIiCSLhQoRERFJFgsVIiIikiwWKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJkkyhsmTJEshkMrz00kumDoWIiIgkQhKFSkpKCj799FN06tTJ1KEQERGRhJi8UCkvL8e4cePw+eefw8PDw9ThEBERkYTYmDqAKVOmYMiQIejfvz8WLlxY57aVlZWorKxUf19SUgIAKCwshEKhaFQcCoUCFRUVKCgogK2tbaP21RxZcv6WnDvA/C05f0vOHbDs/E2de1lZGQBACFHvtiYtVBITE3HkyBGkpKQ0aPvFixdj/vz5WstbtWpl6NCIiIjIyMrKyuDm5lbnNjLRkHLGCDIzMxEdHY2kpCT1syn3338/unTpgoSEBJ2vufOKikqlQmFhITw9PSGTyRoVT2lpKYKCgpCZmQlXV9dG7as5suT8LTl3gPlbcv6WnDtg2fmbOnchBMrKyhAQEAArq7qfQjHZFZXDhw8jLy8PXbt2VS9TKpX4/fff8eGHH6KyshLW1tYar5HL5ZDL5RrL3N3dDRqXq6urxX1gb2fJ+Vty7gDzt+T8LTl3wLLzN2Xu9V1JqWGyQqVfv344duyYxrKJEyeibdu2ePXVV7WKFCIiIrI8JitUXFxc0KFDB41lTk5O8PT01FpORERElsnkw5OlQi6XY+7cuVq3liyFJedvybkDzN+S87fk3AHLzr855W6yh2mJiIiI6sMrKkRERCRZLFSIiIhIslioEBERkWSxUCEiIiLJaraFyuLFixETEwMXFxf4+Phg+PDhOH36tMY2999/P2QymcbXc889p7FNRkYGhgwZAkdHR/j4+GDmzJmorq7W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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[6.30165767]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0oG7TbR04i2U",
        "outputId": "8e57e93e-1f70-4a41-9005-eabbb80ebb63"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          },
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[[6.30165767]]\n"
          ]
        }
      ],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "# Download and prepare the data\n",
        "data_root = \"https://github.com/ageron/data/raw/main/\"\n",
        "lifesat = pd.read_csv(data_root + \"lifesat/lifesat.csv\")\n",
        "X = lifesat[[\"GDP per capita (USD)\"]].values\n",
        "y = lifesat[[\"Life satisfaction\"]].values\n",
        "\n",
        "# Visualize the data\n",
        "lifesat.plot(kind='scatter', grid=True,\n",
        "             x=\"GDP per capita (USD)\", y=\"Life satisfaction\")\n",
        "plt.axis([23_500, 62_500, 4, 9])\n",
        "plt.show()\n",
        "\n",
        "# Select a linear model\n",
        "model = LinearRegression()\n",
        "\n",
        "# Train the model\n",
        "model.fit(X, y)\n",
        "\n",
        "# Make a prediction for Cyprus\n",
        "X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020\n",
        "print(model.predict(X_new)) # outputs [[6.30165767]]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "D0LXz-Zf4i2U"
      },
      "source": [
        "Replacing the Linear Regression model with k-Nearest Neighbors (in this example, k = 3) regression in the previous code is as simple as replacing these two\n",
        "lines:\n",
        "\n",
        "```python\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "model = LinearRegression()\n",
        "```\n",
        "\n",
        "with these two:\n",
        "\n",
        "```python\n",
        "from sklearn.neighbors import KNeighborsRegressor\n",
        "\n",
        "model = KNeighborsRegressor(n_neighbors=3)\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SY54jk4K4i2U",
        "outputId": "a5d3fd91-0639-45a5-bc09-765673f88d12"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[[6.33333333]]\n"
          ]
        }
      ],
      "source": [
        "# Select a 3-Nearest Neighbors regression model\n",
        "from sklearn.neighbors import KNeighborsRegressor\n",
        "\n",
        "model = KNeighborsRegressor(n_neighbors=3)\n",
        "\n",
        "# Train the model\n",
        "model.fit(X, y)\n",
        "\n",
        "# Make a prediction for Cyprus\n",
        "print(model.predict(X_new)) # outputs [[6.33333333]]\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "le6H3S0y4i2V"
      },
      "source": [
        "# Generating the data and figures — please skip"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ORXkYYaD4i2V"
      },
      "source": [
        "This is the code I used to generate the `lifesat.csv` dataset. You can safely skip this."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UrXNspu24i2V"
      },
      "source": [
        "Create a function to save the figures:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {

exec
/bin/bash -lc "sed -n '830,930p' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
      "metadata": {
        "id": "TgWRnlB-4i2V"
      },
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "\n",
        "# Where to save the figures\n",
        "IMAGES_PATH = Path() / \"images\" / \"fundamentals\"\n",
        "IMAGES_PATH.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "def save_fig(fig_id, tight_layout=True, fig_extension=\"png\", resolution=300):\n",
        "    path = IMAGES_PATH / f\"{fig_id}.{fig_extension}\"\n",
        "    if tight_layout:\n",
        "        plt.tight_layout()\n",
        "    plt.savefig(path, format=fig_extension, dpi=resolution)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QQFUuQJh4i2V"
      },
      "source": [
        "## Load and prepare Life satisfaction data"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AQeVJKqd4i2V"
      },
      "source": [
        "To create `lifesat.csv`, I downloaded the Better Life Index (BLI) data from [OECD's website](http://stats.oecd.org/index.aspx?DataSetCode=BLI) (to get the Life Satisfaction for each country), and World Bank GDP per capita data from [OurWorldInData.org](https://ourworldindata.org/grapher/gdp-per-capita-worldbank). The BLI data is in `datasets/lifesat/oecd_bli.csv` (data from 2020), and the GDP per capita data is in `datasets/lifesat/gdp_per_capita.csv` (data up to 2020).\n",
        "\n",
        "If you want to grab the latest versions, please feel free to do so. However, there may be some changes (e.g., in the column names, or different countries missing data), so be prepared to have to tweak the code."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1zHty5254i2V",
        "outputId": "3f64026e-48c3-400f-8505-1665a3fbd4c3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Downloading oecd_bli.csv\n",
            "Downloading gdp_per_capita.csv\n"
          ]
        }
      ],
      "source": [
        "import urllib.request\n",
        "\n",
        "datapath = Path() / \"datasets\" / \"lifesat\"\n",
        "datapath.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "data_root = \"https://github.com/ageron/data/raw/main/\"\n",
        "for filename in (\"oecd_bli.csv\", \"gdp_per_capita.csv\"):\n",
        "    if not (datapath / filename).is_file():\n",
        "        print(\"Downloading\", filename)\n",
        "        url = data_root + \"lifesat/\" + filename\n",
        "        urllib.request.urlretrieve(url, datapath / filename)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "huQBvCbq4i2V"
      },
      "outputs": [],
      "source": [
        "oecd_bli = pd.read_csv(datapath / \"oecd_bli.csv\")\n",
        "gdp_per_capita = pd.read_csv(datapath / \"gdp_per_capita.csv\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-ClCZCyl4i2V"
      },
      "source": [
        "Preprocess the GDP per capita data to keep only the year 2020:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vWwjQ0vG4i2V",
        "outputId": "35589d8d-015f-47b3-f746-dd8f18043686"
      },
      "outputs": [
        {
          "data": {
            "text/html": [

codex
I’ve got enough notebook context. The intro will frame this chapter around the GDP/life-satisfaction example, the idea of learning from data instead of hand-written rules, model choice, and where this orientation chapter sits before the more technical chapters.
apply patch
patch: completed
/home/jevert/git/handson-ml3/01_the_machine_learning_landscape.ipynb
diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..36df23bc2049404374b3dfca7e7e25f32970f2f4
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:**** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.**\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..36df23bc2049404374b3dfca7e7e25f32970f2f4
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:**** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.**\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

codex
The notebook has been patched. I’m doing a quick validation pass now: JSON parses, cell 0 is still markdown, and only the intro text changed.
exec
/bin/bash -lc "python - <<'PY'
import json
from pathlib import Path
p = Path('01_the_machine_learning_landscape.ipynb')
nb = json.loads(p.read_text())
print('cells', len(nb['cells']))
print('cell0_type', nb['cells'][0]['cell_type'])
print('cell0_words', sum(len(line.split()) for line in nb['cells'][0]['source']))
print('cell0_preview')
print(''.join(nb['cells'][0]['source'][:8]))
PY" in /home/jevert/git/handson-ml3
 exited 127 in 0ms:
/bin/bash: line 1: python: command not found

exec
/bin/bash -lc "git diff --stat -- 01_the_machine_learning_landscape.ipynb && git diff --unified=8 -- 01_the_machine_learning_landscape.ipynb | sed -n '1,120p'" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
 01_the_machine_learning_landscape.ipynb | 20 ++++++++++++++++----
 1 file changed, 16 insertions(+), 4 deletions(-)
diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b..36df23b 100644
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -1,21 +1,33 @@
 {
   "cells": [
     {
       "cell_type": "markdown",
       "metadata": {
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
+        "\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:**** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.**\n",
+        "\n",
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
       "cell_type": "markdown",
       "metadata": {
         "id": "gZ1qzcpg4i2T"
       },
       "source": [
@@ -2404,9 +2416,9 @@
       "width": "213px"
     },
     "colab": {
       "provenance": []
     }
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..36df23bc2049404374b3dfca7e7e25f32970f2f4
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:**** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.**\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

codex
The patch is structurally isolated to cell 0. I caught one markdown typo in the vocabulary line, so I’m correcting that now and re-checking with `jq` instead of Python.
apply patch
patch: completed
/home/jevert/git/handson-ml3/01_the_machine_learning_landscape.ipynb
diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..38f320d86d8af40cbe5a73bfd076c24b9d21bc31
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..38f320d86d8af40cbe5a73bfd076c24b9d21bc31
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

exec
/bin/bash -lc "jq -r '.cells[0].cell_type, (.cells[0].source | join(\"\"))' 01_the_machine_learning_landscape.ipynb | sed -n '1,80p'" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
markdown
## Chapter Overview: The Machine Learning Landscape

This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.

**What you should walk away understanding:**

1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.
2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.
3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.
4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.
5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.

This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.

**Key vocabulary to know before diving in:** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.

One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use.

exec
/bin/bash -lc "jq '.cells | length' 01_the_machine_learning_landscape.ipynb && jq -r '[.cells[0].source[]] | join(\"\") | split(\" \") | length' 01_the_machine_learning_landscape.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
53
403

diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..38f320d86d8af40cbe5a73bfd076c24b9d21bc31
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

codex
Replaced cell 0 in [01_the_machine_learning_landscape.ipynb](/home/jevert/git/handson-ml3/01_the_machine_learning_landscape.ipynb) with a 403-word chapter intro that covers the chapter purpose, five learning objectives, why the topic matters in the broader ML landscape, where it fits in the course, and key vocabulary.

Validation passed: the notebook still parses as valid JSON, cell 0 remains a `markdown` cell, the notebook still has 53 cells, and no code cells were edited.
diff --git a/01_the_machine_learning_landscape.ipynb b/01_the_machine_learning_landscape.ipynb
index 217ea4b11adb2eb7b9b2015a7bb694955be59dff..38f320d86d8af40cbe5a73bfd076c24b9d21bc31
--- a/01_the_machine_learning_landscape.ipynb
+++ b/01_the_machine_learning_landscape.ipynb
@@ -6,11 +6,23 @@
         "id": "Zs3Na-QS4i2S"
       },
       "source": [
-        "**Chapter 1 – The Machine Learning landscape**\n",
+        "## Chapter Overview: The Machine Learning Landscape\n",
+        "\n",
+        "This chapter is an orientation to how machine learning fits together as a field. Rather than starting with a long catalog of algorithms, it begins with the central idea that makes machine learning useful: instead of hard-coding rules for every case, we let a model learn patterns from data and then use those patterns to make predictions on new examples. The notebook's GDP-per-capita versus life-satisfaction example is deliberately simple, because the point is not mathematical sophistication yet. The point is to build intuition for what training data, a model, a prediction, and a learning algorithm actually do in practice.\n",
+        "\n",
+        "**What you should walk away understanding:**\n",
+        "\n",
+        "1. **What machine learning is solving**: when it is better to learn from data than to write explicit rules by hand, and why this matters for messy real-world problems.\n",
+        "2. **The supervised learning workflow**: inputs (features), outputs (targets), fitting a model on examples, and then using that fitted model to make predictions for unseen cases.\n",
+        "3. **Generalization as the real goal**: a model is useful only if it performs well beyond the examples it was trained on, not just because it fits the training data.\n",
+        "4. **Why model choice matters**: even in this notebook, a linear regression model and a k-nearest neighbors regressor make slightly different assumptions and can produce different predictions from the same data.\n",
+        "5. **Why data matters as much as algorithms**: collecting, cleaning, aligning, and visualizing data is a major part of ML work, and much of this notebook's later material reflects that reality.\n",
         "\n",
-        "_This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._\n",
+        "This topic matters because it gives you the map before you start navigating the territory. Later chapters will go deeper into training, optimization, evaluation, regularization, and specific model families, but those ideas are much easier to place once you understand the broad ML landscape first. Chapter 1 sits before the technical deep dive: nothing important comes before it in the book, and almost everything that comes next depends on the distinctions introduced here.\n",
+        "\n",
+        "**Key vocabulary to know before diving in:** model, feature, target, training set, prediction, supervised learning, regression, learning algorithm, generalization, and instance-based versus model-based learning.\n",
         "\n",
-        "You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter."
+        "One practical note: this notebook contains both the simple chapter example and additional data-preparation code used to generate `lifesat.csv` and reproduce some figures. The conceptual lesson is the main attraction. The extra plumbing is useful if you want to see where the dataset comes from and how raw source data becomes something a model can actually use."
       ]
     },
     {
@@ -2409,4 +2421,4 @@
   },
   "nbformat": 4,
   "nbformat_minor": 0
-}
\ No newline at end of file
+}

tokens used
69,549
```
