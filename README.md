# MNIST Digit Classifier — From-Scratch Perceptron + Drawing GUI

A single-layer neural network (softmax regression / perceptron) for
classifying handwritten digits from the MNIST dataset, implemented from
scratch with NumPy (no deep learning framework used for the model itself).
Includes training visualizations, model persistence, and a desktop GUI that
lets you draw a digit with the mouse and see the model's prediction live.

## Project overview

- **Model**: single dense layer, softmax activation, categorical
  cross-entropy loss, trained with gradient descent (both a pure per-sample
  SGD version and a vectorized mini-batch version are included).
- **Training data**: MNIST (60,000 train / 10,000 test images, 28x28
  grayscale), loaded via `keras.datasets.mnist`.
- **GUI**: a Tkinter app where you draw a digit and see the model's
  prediction, the full softmax probability distribution, and a live preview
  of exactly what the model receives as input (28x28), updated as you draw.
- **Data collection**: the GUI also lets you explicitly save your own
  drawings (with a label you choose) to expand the training set beyond the
  original MNIST scans.

## Requirements

- Python 3.10+
- See `requirements.txt` for exact packages.

## Setup (on a new machine)

```bash
# 1. Clone / copy the project folder, then move into it
cd path/to/project

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The first time you load MNIST (`mnist.load_data()`), Keras downloads the
dataset automatically (~11 MB) and caches it in `~/.keras/datasets/` — you
need an internet connection for that one-time download.

## Project files

| File | Purpose |
|---|---|
| `*.ipynb` / main notebook | Full training pipeline: data loading, model definition (`NeuralNetwork` class), training loop, loss/accuracy visualization, evaluation, confusion matrix. |
| `mini_batch_sgd.py`, `mini_batch_sgd_multi_epoch.py` | Vectorized mini-batch training variant (faster, smoother loss curve) — meant to be pasted as extra cells into the main notebook. |
| `trening_per_krok_v2.py` | Per-step (not per-epoch) validation tracking variant. |
| `zapis_modelu_finalny.py` | `save_model()` / `load_model()` — persist trained weights to a `.npz` file. |
| `podglad_collected_data.py` | Notebook snippet to preview digits saved via the GUI's data-collection feature. |
| `gui_rozpoznawanie_cyfr.py` | **Standalone desktop app** — draw a digit, see live predictions, save labeled samples. Run with `python`, not as a notebook cell. |
| `kalibrator_pedzla.py` | Standalone tool to visually tune the drawing brush's softness parameters against real MNIST samples. |
| `mnist_samples.npz` | Small set of real MNIST images bundled for the calibration tool (must sit next to `kalibrator_pedzla.py`). |
| `requirements.txt` | Python dependencies. |

> The notebook files here use the `#%%` cell-marker format (Jupytext "light"
> script format). You can open them as a notebook in **PyCharm Professional**,
> **VS Code** (with the Python extension — it recognizes `# %%` cells), or
> convert to a classic `.ipynb` with `jupytext --to notebook your_file.py`.

## Usage

### 1. Train the model

Open the main notebook and run all cells top to bottom. This will:
- download/load MNIST,
- train the model (`NN`, per-sample SGD) and/or `NN_batch` (mini-batch),
- show loss/accuracy curves and a confusion matrix on the test set.

### 2. Save the trained model

Run the `save_model()` cell — this writes a `.npz` file (weights, biases,
layer sizes) to a `models/` folder, e.g.:

```python
save_model(NN, "models/mnist_perceptron.npz")
```

### 3. Run the drawing GUI

```bash
python gui_rozpoznawanie_cyfr.py
```

Run it **from the project's root folder** (or wherever your `models/`
folder lives) so it can auto-load the most recent saved model. You can also
load a specific model manually via the "Wczytaj model..." button.

Controls:
- Draw with the left mouse button.
- Prediction updates live as you draw — no button needed.
- `F11` — toggle fullscreen, `Esc` — exit fullscreen.
- "Wyczysc" — clear the canvas.
- Digit buttons (0–9) + "Zapisz jako cyfra X" — explicitly save the current
  drawing as a labeled training sample to `collected_data/`.

### 4. (Optional) Tune the brush softness

```bash
python kalibrator_pedzla.py
```

Requires `mnist_samples.npz` in the same folder. Lets you compare your
drawn digits against real MNIST examples side by side while adjusting the
brush's `core_radius` / `sigma` sliders, so the model sees input that looks
like what it was trained on.

### 5. (Optional) Extend the training set with your own drawings

After collecting samples via the GUI, load them back in the notebook:

```python
import numpy as np

data = np.load("collected_data/collected_dataset.npz")
x_collected, y_collected = data["images"], data["labels"]

x_train_extended = np.vstack([x_train, x_collected])
y_train_extended = np.concatenate([y_train, y_collected])
```

## Notes for running on a different machine

- Paths in the scripts (`models/`, `collected_data/`) are **relative** —
  always run the scripts from the project's root folder, or the auto-load /
  auto-save features won't find the right files.
- The GUI uses Tkinter, which ships with standard Python on Windows/macOS.
  On some Linux distributions you may need to install it separately, e.g.
  `sudo apt install python3-tk`.
- No GPU is required — this is a from-scratch NumPy implementation and
  trains fine on CPU.
