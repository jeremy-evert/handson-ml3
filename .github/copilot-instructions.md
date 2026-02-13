# Copilot Instructions for Hands-On Machine Learning (3rd Edition)

This is an educational Jupyter notebook repository accompanying "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)". It contains 19 chapter notebooks covering ML fundamentals through reinforcement learning, plus supporting tutorials.

## Repository Structure

- **Main chapters**: `01_*.ipynb` through `19_*.ipynb` (~19 notebooks)
- **Tools tutorials**: `tools_numpy.ipynb`, `tools_matplotlib.ipynb`, `tools_pandas.ipynb`
- **Math tutorials**: `math_linear_algebra.ipynb`, `math_differential_calculus.ipynb`
- **Extra content**: `extra_autodiff.ipynb`, `extra_gradient_descent_comparison.ipynb`, etc.
- **Infrastructure**: `environment.yml` (conda), `requirements.txt` (pip), `INSTALL.md`, `docker/` setup

## Python Environment & Setup

- **Python**: 3.10 recommended, 3.7+ required (enforced with `assert sys.version_info >= (3, 7)`)
- **Conda environment**: `homl3` - created from `environment.yml`
- **Kernel registration**: `python3 ipykernel install --user --name=python3` (name matters for notebook compatibility)
- **GPU support**: Optional but documented (CUDA/cuDNN required for TensorFlow GPU)

**Key libraries**: scikit-learn 1.3+, TensorFlow 2.14, Keras 2/3, NumPy 1.26, Pandas 2.1, Matplotlib 3.8

## Notebook Conventions

1. **Cell structure**: Markdown cells for explanations, code cells for executable examples
2. **Version assertions**: First code cells validate Python and package versions
3. **Random seeding**: Use `np.random.seed(42)` for reproducibility (critical for consistency across runs)
4. **Comment markers**: `# extra code –` marks supplementary content not in the published book
5. **Helper functions**: Define reusable utilities early (e.g., `load_housing_data()`, `save_fig()`)
6. **Imports**: Follow Python standard → third-party → local pattern; matplotlib/numpy imported as `plt`/`np`

## Common Patterns

### Data Loading

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))
```

Data typically downloaded from `https://github.com/ageron/data/` or via TensorFlow datasets API.

### Figure Saving

```python
IMAGES_PATH = Path() / "images" / "chapter_name"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

### Model Training Pattern

1. Load/prepare data with train/test split (stratified when appropriate)
2. Train multiple models with `random_state=42`
3. Evaluate with cross-validation or test set
4. Hyperparameter tuning with keras-tuner (chapters 10+)
5. Visualize results with Matplotlib

## Key Dependencies & Usage

| Library | Version | Primary Use |
|---------|---------|------------|
| scikit-learn | 1.3 | Classical ML (trees, linear, ensemble) |
| TensorFlow | 2.14 | Deep learning, Keras API |
| NumPy | 1.26 | Numerical operations, arrays |
| Pandas | 2.1 | Data manipulation, CSV loading |
| Matplotlib | 3.8 | Visualization, figure saving |
| XGBoost | 2.0 | Gradient boosting (ch7) |
| Transformers | 4.35 | NLP models (ch16) |
| Gymnasium | 0.29 | Reinforcement learning (ch18) |

## Reproducibility Conventions

- Use `random_state=` parameter in all stochastic operations
- Set `np.random.seed(42)` before split/sampling operations
- Hardcode test ratios (typically 0.2) for train/test splits
- Use stratified splitting when classes are imbalanced

## GPU & Performance Notes

- TensorFlow uses GPU automatically if available (CUDA-capable NVIDIA GPU required)
- Fallback to CPU is silent; inspect training speed to detect GPU usage
- Some operations (e.g., `tf.function`) require special attention for graph compilation
- Chapter 18 (RL) requires `TF_USE_LEGACY_KERAS=1` environment variable due to Keras 3 compatibility

## Multi-Environment Support

- Notebooks designed for Jupyter Lab/Notebook, Google Colab, Kaggle Notebooks
- Colab badges in notebooks link directly to Colab execution
- Code works identically across environments (paths, imports)
- `PYTHONHASHSEED=0` environment variable ensures hash-based ordering reproducibility

## Editing Guidelines

- Preserve chapter structure: explanations before code, progressive complexity
- Mark new code with `# extra code –` comment if supplementary to book content
- Use consistent `random_state=42` across all new ML code
- Save high-res figures (300 dpi) to `images/` subdirectories
- Test on both CPU and GPU environments if adding TensorFlow code
