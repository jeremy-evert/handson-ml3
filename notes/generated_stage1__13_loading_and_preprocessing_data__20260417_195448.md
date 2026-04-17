# generated_stage1__13_loading_and_preprocessing_data__20260417_195448

- run_id: `generated_stage1__13_loading_and_preprocessing_data__20260417_195448`
- prompt_file: `codex_prompts/generated_stage1__13_loading_and_preprocessing_data.md`
- prompt_stem: `generated_stage1__13_loading_and_preprocessing_data`
- started_at_utc: `20260417_195448`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260417_195533`
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

- elapsed_seconds: `44.562`
- final_output_char_count: `445`
- stderr_char_count: `29077`

## Prompt Text

```md
# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: 13_loading_and_preprocessing_data.ipynb
- Chapter: 13
- Notebook stem: 13_loading_and_preprocessing_data

## Current Chapter Intro State
Status: MISSING — no chapter intro exists. Insert a new one.

Current content (if any):
````

(no existing intro)

```

## Your Task
INSERT a new markdown cell at index 3. Place it after the Colab/Kaggle links table and before the # Setup cell.

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

1. Read the full notebook from: `13_loading_and_preprocessing_data.ipynb`
2. INSERT a new markdown cell at index 3. Place it after the Colab/Kaggle links table and before the # Setup cell.
3. Write the COMPLETE modified notebook as valid JSON back to: `13_loading_and_preprocessing_data.ipynb`
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

Inserted a new markdown intro cell at index `3` in [13_loading_and_preprocessing_data.ipynb](/home/jevert/git/handson-ml3/13_loading_and_preprocessing_data.ipynb). The existing Colab/Kaggle links cell remains immediately before it, and the `# Setup` markdown cell is now at index `4`.

Validation checks passed: the notebook is valid JSON, the new cell is `markdown` with a `source` field, and no code cells were modified, removed, or reordered.

## Stderr

```text
Runner context:
- sandbox: workspace-write
- full_auto: False
- model: (default)
- profile: (default)
- json: False
- codex_command: codex exec -C /home/jevert/git/handson-ml3 --sandbox workspace-write --output-last-message /tmp/codex-last-message-pego3q8o.txt -
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /home/jevert/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d9d02-a38a-7861-8cef-12f4ce3b6267
--------
user
# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: 13_loading_and_preprocessing_data.ipynb
- Chapter: 13
- Notebook stem: 13_loading_and_preprocessing_data

## Current Chapter Intro State
Status: MISSING — no chapter intro exists. Insert a new one.

Current content (if any):
````

(no existing intro)

```

## Your Task
INSERT a new markdown cell at index 3. Place it after the Colab/Kaggle links table and before the # Setup cell.

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

1. Read the full notebook from: `13_loading_and_preprocessing_data.ipynb`
2. INSERT a new markdown cell at index 3. Place it after the Colab/Kaggle links table and before the # Setup cell.
3. Write the COMPLETE modified notebook as valid JSON back to: `13_loading_and_preprocessing_data.ipynb`
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
I’m opening the notebook structure first to confirm the existing cell order and exact insertion point at cell index 3, then I’ll patch only that markdown addition and leave every code cell untouched.
exec
/bin/bash -lc "jq '.cells[0:6] | map({cell_type, source})' 13_loading_and_preprocessing_data.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
[
  {
    "cell_type": "markdown",
    "source": [
      "**Chapter 13 – Loading and Preprocessing Data with TensorFlow**"
    ]
  },
  {
    "cell_type": "markdown",
    "source": [
      "_This notebook contains all the sample code and solutions to the exercises in chapter 13._"
    ]
  },
  {
    "cell_type": "markdown",
    "source": [
      "<table align=\"left\">\n",
      "  <td>\n",
      "    <a href=\"https://colab.research.google.com/github/ageron/handson-ml3/blob/main/13_loading_and_preprocessing_data.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
      "  </td>\n",
      "  <td>\n",
      "    <a target=\"_blank\" href=\"https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/13_loading_and_preprocessing_data.ipynb\"><img src=\"https://kaggle.com/static/images/open-in-kaggle.svg\" /></a>\n",
      "  </td>\n",
      "</table>"
    ]
  },
  {
    "cell_type": "markdown",
    "source": [
      "# Setup"
    ]
  },
  {
    "cell_type": "markdown",
    "source": [
      "This project requires Python 3.7 or above:"
    ]
  },
  {
    "cell_type": "code",
    "source": [
      "import sys\n",
      "\n",
      "assert sys.version_info >= (3, 7)"
    ]
  }
]

exec
/bin/bash -lc 'git status --short 13_loading_and_preprocessing_data.ipynb' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
codex
I confirmed the insertion point: the notebook currently goes title, subtitle, links table, then `# Setup`. I’m reading the raw JSON around that boundary now so I can splice in one markdown cell without perturbing any neighboring metadata or code cells.
exec
/bin/bash -lc "rg -n '\"\\*\\*Chapter 13|# Setup|</table>' 13_loading_and_preprocessing_data.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
7:    "**Chapter 13 – Loading and Preprocessing Data with TensorFlow**"
28:    "</table>"
37:    "# Setup"

exec
/bin/bash -lc "sed -n '1,80p' 13_loading_and_preprocessing_data.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Chapter 13 – Loading and Preprocessing Data with TensorFlow**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "_This notebook contains all the sample code and solutions to the exercises in chapter 13._"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<table align=\"left\">\n",
    "  <td>\n",
    "    <a href=\"https://colab.research.google.com/github/ageron/handson-ml3/blob/main/13_loading_and_preprocessing_data.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/13_loading_and_preprocessing_data.ipynb\"><img src=\"https://kaggle.com/static/images/open-in-kaggle.svg\" /></a>\n",
    "  </td>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Setup"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This project requires Python 3.7 or above:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "\n",
    "assert sys.version_info >= (3, 7)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It also requires Scikit-Learn ≥ 1.0.1:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
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
   "metadata": {},
   "source": [

codex
I have the exact JSON splice point. Next step is a single patch that inserts the new chapter overview cell with the required scope: chapter purpose, 3–5 learning objectives, broader ML relevance, course placement, and key vocabulary.
apply patch
patch: completed
/home/jevert/git/handson-ml3/13_loading_and_preprocessing_data.ipynb
diff --git a/13_loading_and_preprocessing_data.ipynb b/13_loading_and_preprocessing_data.ipynb
index 0d531b4b251a3f51d7d6cb107684e9e4f36b84c6..3a2b323cdc8175d95862ed5dcb09eb990eb63b3f
--- a/13_loading_and_preprocessing_data.ipynb
+++ b/13_loading_and_preprocessing_data.ipynb
@@ -30,6 +30,29 @@
   },
   {
    "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Chapter Overview: Loading and Preprocessing Data with TensorFlow\n",
+    "\n",
+    "This chapter is about the part of machine learning work that often determines whether a model is usable at all: getting data from storage into a training pipeline efficiently, consistently, and at scale. Up to this point in the book, most datasets have been small enough to fit comfortably in memory and simple enough to manipulate with familiar tools. Real production systems are different. Data may live across many files, arrive as serialized records, require parsing and transformation on the fly, and need to be streamed quickly enough to keep accelerators busy rather than waiting on I/O.\n",
+    "\n",
+    "**What you should walk away understanding:**\n",
+    "\n",
+    "1. **Why input pipelines matter**: model quality is only part of the story; slow, fragile, or inconsistent data loading can bottleneck the entire training process.\n",
+    "2. **How TensorFlow's `tf.data` API works**: building datasets from tensors or files, chaining transformations, batching, shuffling, repeating, and prefetching data.\n",
+    "3. **How preprocessing moves into the model pipeline**: using TensorFlow tools to normalize, encode, bucketize, and otherwise transform raw features in a reproducible way.\n",
+    "4. **How TFRecord and serialization fit into scalable ML**: storing examples in a format optimized for TensorFlow training workloads.\n",
+    "5. **What tradeoffs appear in practice**: memory use versus throughput, randomness versus reproducibility, and convenience versus production-ready input design.\n",
+    "\n",
+    "This topic matters in the broader ML landscape because data pipelines are where experimentation meets engineering reality. Many training problems that look like model issues are actually data problems: corrupted examples, inconsistent preprocessing between training and serving, poor shuffling, or pipelines that starve the GPU. A solid preprocessing and loading workflow improves reliability, reproducibility, and training speed, all of which matter just as much in research notebooks as in deployed systems.\n",
+    "\n",
+    "In the flow of the book, this chapter sits between core model-building ideas and more advanced deep learning practice. Earlier chapters focused on selecting models, training them, evaluating them, and tuning them with classical ML workflows. Here, the emphasis shifts toward infrastructure for deep learning workloads, where dataset size and throughput become first-class concerns. What comes next builds on this foundation: once you can feed data cleanly and efficiently into TensorFlow, you are in a much better position to train larger neural networks and work with more realistic pipelines.\n",
+    "\n",
+    "Key vocabulary to know before diving in: **input pipeline**, **dataset**, **batch**, **shuffle buffer**, **prefetching**, **serialization**, **TFRecord**, **parsing**, **feature preprocessing**, and **data transformation**. These terms show up repeatedly because this chapter is really about turning raw data into a dependable stream of model-ready examples.\n"
+   ]
+  },
+  {
+   "cell_type": "markdown",
    "metadata": {
     "tags": []
    },

diff --git a/13_loading_and_preprocessing_data.ipynb b/13_loading_and_preprocessing_data.ipynb
index 0d531b4b251a3f51d7d6cb107684e9e4f36b84c6..3a2b323cdc8175d95862ed5dcb09eb990eb63b3f
--- a/13_loading_and_preprocessing_data.ipynb
+++ b/13_loading_and_preprocessing_data.ipynb
@@ -30,6 +30,29 @@
   },
   {
    "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Chapter Overview: Loading and Preprocessing Data with TensorFlow\n",
+    "\n",
+    "This chapter is about the part of machine learning work that often determines whether a model is usable at all: getting data from storage into a training pipeline efficiently, consistently, and at scale. Up to this point in the book, most datasets have been small enough to fit comfortably in memory and simple enough to manipulate with familiar tools. Real production systems are different. Data may live across many files, arrive as serialized records, require parsing and transformation on the fly, and need to be streamed quickly enough to keep accelerators busy rather than waiting on I/O.\n",
+    "\n",
+    "**What you should walk away understanding:**\n",
+    "\n",
+    "1. **Why input pipelines matter**: model quality is only part of the story; slow, fragile, or inconsistent data loading can bottleneck the entire training process.\n",
+    "2. **How TensorFlow's `tf.data` API works**: building datasets from tensors or files, chaining transformations, batching, shuffling, repeating, and prefetching data.\n",
+    "3. **How preprocessing moves into the model pipeline**: using TensorFlow tools to normalize, encode, bucketize, and otherwise transform raw features in a reproducible way.\n",
+    "4. **How TFRecord and serialization fit into scalable ML**: storing examples in a format optimized for TensorFlow training workloads.\n",
+    "5. **What tradeoffs appear in practice**: memory use versus throughput, randomness versus reproducibility, and convenience versus production-ready input design.\n",
+    "\n",
+    "This topic matters in the broader ML landscape because data pipelines are where experimentation meets engineering reality. Many training problems that look like model issues are actually data problems: corrupted examples, inconsistent preprocessing between training and serving, poor shuffling, or pipelines that starve the GPU. A solid preprocessing and loading workflow improves reliability, reproducibility, and training speed, all of which matter just as much in research notebooks as in deployed systems.\n",
+    "\n",
+    "In the flow of the book, this chapter sits between core model-building ideas and more advanced deep learning practice. Earlier chapters focused on selecting models, training them, evaluating them, and tuning them with classical ML workflows. Here, the emphasis shifts toward infrastructure for deep learning workloads, where dataset size and throughput become first-class concerns. What comes next builds on this foundation: once you can feed data cleanly and efficiently into TensorFlow, you are in a much better position to train larger neural networks and work with more realistic pipelines.\n",
+    "\n",
+    "Key vocabulary to know before diving in: **input pipeline**, **dataset**, **batch**, **shuffle buffer**, **prefetching**, **serialization**, **TFRecord**, **parsing**, **feature preprocessing**, and **data transformation**. These terms show up repeatedly because this chapter is really about turning raw data into a dependable stream of model-ready examples.\n"
+   ]
+  },
+  {
+   "cell_type": "markdown",
    "metadata": {
     "tags": []
    },

codex
The insertion is in place. I’m validating that the notebook is still valid JSON and that cell index 3 is now the new markdown intro, with no code-cell changes or cell reordering beyond that single insertion.
exec
/bin/bash -lc "jq '.cells[3] | {cell_type, metadata, source}' 13_loading_and_preprocessing_data.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
{
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "## Chapter Overview: Loading and Preprocessing Data with TensorFlow\n",
    "\n",
    "This chapter is about the part of machine learning work that often determines whether a model is usable at all: getting data from storage into a training pipeline efficiently, consistently, and at scale. Up to this point in the book, most datasets have been small enough to fit comfortably in memory and simple enough to manipulate with familiar tools. Real production systems are different. Data may live across many files, arrive as serialized records, require parsing and transformation on the fly, and need to be streamed quickly enough to keep accelerators busy rather than waiting on I/O.\n",
    "\n",
    "**What you should walk away understanding:**\n",
    "\n",
    "1. **Why input pipelines matter**: model quality is only part of the story; slow, fragile, or inconsistent data loading can bottleneck the entire training process.\n",
    "2. **How TensorFlow's `tf.data` API works**: building datasets from tensors or files, chaining transformations, batching, shuffling, repeating, and prefetching data.\n",
    "3. **How preprocessing moves into the model pipeline**: using TensorFlow tools to normalize, encode, bucketize, and otherwise transform raw features in a reproducible way.\n",
    "4. **How TFRecord and serialization fit into scalable ML**: storing examples in a format optimized for TensorFlow training workloads.\n",
    "5. **What tradeoffs appear in practice**: memory use versus throughput, randomness versus reproducibility, and convenience versus production-ready input design.\n",
    "\n",
    "This topic matters in the broader ML landscape because data pipelines are where experimentation meets engineering reality. Many training problems that look like model issues are actually data problems: corrupted examples, inconsistent preprocessing between training and serving, poor shuffling, or pipelines that starve the GPU. A solid preprocessing and loading workflow improves reliability, reproducibility, and training speed, all of which matter just as much in research notebooks as in deployed systems.\n",
    "\n",
    "In the flow of the book, this chapter sits between core model-building ideas and more advanced deep learning practice. Earlier chapters focused on selecting models, training them, evaluating them, and tuning them with classical ML workflows. Here, the emphasis shifts toward infrastructure for deep learning workloads, where dataset size and throughput become first-class concerns. What comes next builds on this foundation: once you can feed data cleanly and efficiently into TensorFlow, you are in a much better position to train larger neural networks and work with more realistic pipelines.\n",
    "\n",
    "Key vocabulary to know before diving in: **input pipeline**, **dataset**, **batch**, **shuffle buffer**, **prefetching**, **serialization**, **TFRecord**, **parsing**, **feature preprocessing**, and **data transformation**. These terms show up repeatedly because this chapter is really about turning raw data into a dependable stream of model-ready examples.\n"
  ]
}

exec
/bin/bash -lc "jq '.cells[4] | {cell_type, source}' 13_loading_and_preprocessing_data.ipynb" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
{
  "cell_type": "markdown",
  "source": [
    "# Setup"
  ]
}

exec
/bin/bash -lc 'jq empty 13_loading_and_preprocessing_data.ipynb' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
diff --git a/13_loading_and_preprocessing_data.ipynb b/13_loading_and_preprocessing_data.ipynb
index 0d531b4b251a3f51d7d6cb107684e9e4f36b84c6..3a2b323cdc8175d95862ed5dcb09eb990eb63b3f
--- a/13_loading_and_preprocessing_data.ipynb
+++ b/13_loading_and_preprocessing_data.ipynb
@@ -30,6 +30,29 @@
   },
   {
    "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Chapter Overview: Loading and Preprocessing Data with TensorFlow\n",
+    "\n",
+    "This chapter is about the part of machine learning work that often determines whether a model is usable at all: getting data from storage into a training pipeline efficiently, consistently, and at scale. Up to this point in the book, most datasets have been small enough to fit comfortably in memory and simple enough to manipulate with familiar tools. Real production systems are different. Data may live across many files, arrive as serialized records, require parsing and transformation on the fly, and need to be streamed quickly enough to keep accelerators busy rather than waiting on I/O.\n",
+    "\n",
+    "**What you should walk away understanding:**\n",
+    "\n",
+    "1. **Why input pipelines matter**: model quality is only part of the story; slow, fragile, or inconsistent data loading can bottleneck the entire training process.\n",
+    "2. **How TensorFlow's `tf.data` API works**: building datasets from tensors or files, chaining transformations, batching, shuffling, repeating, and prefetching data.\n",
+    "3. **How preprocessing moves into the model pipeline**: using TensorFlow tools to normalize, encode, bucketize, and otherwise transform raw features in a reproducible way.\n",
+    "4. **How TFRecord and serialization fit into scalable ML**: storing examples in a format optimized for TensorFlow training workloads.\n",
+    "5. **What tradeoffs appear in practice**: memory use versus throughput, randomness versus reproducibility, and convenience versus production-ready input design.\n",
+    "\n",
+    "This topic matters in the broader ML landscape because data pipelines are where experimentation meets engineering reality. Many training problems that look like model issues are actually data problems: corrupted examples, inconsistent preprocessing between training and serving, poor shuffling, or pipelines that starve the GPU. A solid preprocessing and loading workflow improves reliability, reproducibility, and training speed, all of which matter just as much in research notebooks as in deployed systems.\n",
+    "\n",
+    "In the flow of the book, this chapter sits between core model-building ideas and more advanced deep learning practice. Earlier chapters focused on selecting models, training them, evaluating them, and tuning them with classical ML workflows. Here, the emphasis shifts toward infrastructure for deep learning workloads, where dataset size and throughput become first-class concerns. What comes next builds on this foundation: once you can feed data cleanly and efficiently into TensorFlow, you are in a much better position to train larger neural networks and work with more realistic pipelines.\n",
+    "\n",
+    "Key vocabulary to know before diving in: **input pipeline**, **dataset**, **batch**, **shuffle buffer**, **prefetching**, **serialization**, **TFRecord**, **parsing**, **feature preprocessing**, and **data transformation**. These terms show up repeatedly because this chapter is really about turning raw data into a dependable stream of model-ready examples.\n"
+   ]
+  },
+  {
+   "cell_type": "markdown",
    "metadata": {
     "tags": []
    },

codex
Inserted a new markdown intro cell at index `3` in [13_loading_and_preprocessing_data.ipynb](/home/jevert/git/handson-ml3/13_loading_and_preprocessing_data.ipynb). The existing Colab/Kaggle links cell remains immediately before it, and the `# Setup` markdown cell is now at index `4`.

Validation checks passed: the notebook is valid JSON, the new cell is `markdown` with a `source` field, and no code cells were modified, removed, or reordered.
diff --git a/13_loading_and_preprocessing_data.ipynb b/13_loading_and_preprocessing_data.ipynb
index 0d531b4b251a3f51d7d6cb107684e9e4f36b84c6..3a2b323cdc8175d95862ed5dcb09eb990eb63b3f
--- a/13_loading_and_preprocessing_data.ipynb
+++ b/13_loading_and_preprocessing_data.ipynb
@@ -30,6 +30,29 @@
   },
   {
    "cell_type": "markdown",
+   "metadata": {},
+   "source": [
+    "## Chapter Overview: Loading and Preprocessing Data with TensorFlow\n",
+    "\n",
+    "This chapter is about the part of machine learning work that often determines whether a model is usable at all: getting data from storage into a training pipeline efficiently, consistently, and at scale. Up to this point in the book, most datasets have been small enough to fit comfortably in memory and simple enough to manipulate with familiar tools. Real production systems are different. Data may live across many files, arrive as serialized records, require parsing and transformation on the fly, and need to be streamed quickly enough to keep accelerators busy rather than waiting on I/O.\n",
+    "\n",
+    "**What you should walk away understanding:**\n",
+    "\n",
+    "1. **Why input pipelines matter**: model quality is only part of the story; slow, fragile, or inconsistent data loading can bottleneck the entire training process.\n",
+    "2. **How TensorFlow's `tf.data` API works**: building datasets from tensors or files, chaining transformations, batching, shuffling, repeating, and prefetching data.\n",
+    "3. **How preprocessing moves into the model pipeline**: using TensorFlow tools to normalize, encode, bucketize, and otherwise transform raw features in a reproducible way.\n",
+    "4. **How TFRecord and serialization fit into scalable ML**: storing examples in a format optimized for TensorFlow training workloads.\n",
+    "5. **What tradeoffs appear in practice**: memory use versus throughput, randomness versus reproducibility, and convenience versus production-ready input design.\n",
+    "\n",
+    "This topic matters in the broader ML landscape because data pipelines are where experimentation meets engineering reality. Many training problems that look like model issues are actually data problems: corrupted examples, inconsistent preprocessing between training and serving, poor shuffling, or pipelines that starve the GPU. A solid preprocessing and loading workflow improves reliability, reproducibility, and training speed, all of which matter just as much in research notebooks as in deployed systems.\n",
+    "\n",
+    "In the flow of the book, this chapter sits between core model-building ideas and more advanced deep learning practice. Earlier chapters focused on selecting models, training them, evaluating them, and tuning them with classical ML workflows. Here, the emphasis shifts toward infrastructure for deep learning workloads, where dataset size and throughput become first-class concerns. What comes next builds on this foundation: once you can feed data cleanly and efficiently into TensorFlow, you are in a much better position to train larger neural networks and work with more realistic pipelines.\n",
+    "\n",
+    "Key vocabulary to know before diving in: **input pipeline**, **dataset**, **batch**, **shuffle buffer**, **prefetching**, **serialization**, **TFRecord**, **parsing**, **feature preprocessing**, and **data transformation**. These terms show up repeatedly because this chapter is really about turning raw data into a dependable stream of model-ready examples.\n"
+   ]
+  },
+  {
+   "cell_type": "markdown",
    "metadata": {
     "tags": []
    },

tokens used
33,356
```
