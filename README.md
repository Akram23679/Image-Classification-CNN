# CIFAR-10 Image Classification with a CNN

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-0f766e?style=for-the-badge)](https://akram23679.github.io/CIFAR10-Image-Classification/)

**Deep Learning · Computer Vision · Akram Ali**

A convolutional neural network (CNN) built from scratch with TensorFlow/Keras that classifies small colour photos into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). It reaches **86.14% accuracy** on the 10,000-image CIFAR-10 test set.

The portfolio webpage includes a **live demo**: upload any photo or pick an example, and the model classifies it instantly in your browser. Unlike the notebook's Gradio link, which expires after a week, the demo stays online permanently.

---

## Live Demo

**[Try it: classify a photo →](https://akram23679.github.io/CIFAR10-Image-Classification/)**

---

## Results at a Glance

| Metric | Value |
|---|---|
| Test accuracy | **86.14%** |
| Test loss | 0.4190 |
| Macro F1 | 0.860 |
| Best class (F1) | Automobile, 0.941 |
| Hardest class (F1) | Cat, 0.740 |
| Model parameters | 1,345,066 |
| Training | 79 epochs (best at 67), Google Colab T4 GPU (~16 s/epoch) |

---

## Dataset

**CIFAR-10** (Krizhevsky, 2009): 60,000 colour images of 32×32 pixels, 6,000 per class.

| Split | Images | Purpose |
|---|---|---|
| Train | 45,000 | Model training (90% of the official training set, stratified) |
| Validation | 5,000 | Early stopping, learning-rate scheduling, checkpointing |
| Test | 10,000 | Final evaluation, used once |

---

## Model Architecture

```
Input (32×32×3)
 → Data augmentation: horizontal flip, rotation ±15°, zoom ±10%, translation ±10% (training only)
 → Block 1: 2 × [Conv2D 3×3, 32, ReLU → BatchNorm] → MaxPool 2×2 → Dropout 0.25    (16×16×32)
 → Block 2: 2 × [Conv2D 3×3, 64, ReLU → BatchNorm] → MaxPool 2×2 → Dropout 0.30    (8×8×64)
 → Block 3: 2 × [Conv2D 3×3, 128, ReLU → BatchNorm] → MaxPool 2×2 → Dropout 0.40   (4×4×128)
 → Flatten (2,048) → Dense 512, ReLU → BatchNorm → Dropout 0.5
 → Dense 10 → Softmax
```

**Training:** Adam (learning rate 0.001), categorical cross-entropy, batch size 64, up to 100 epochs.
**Callbacks:** EarlyStopping (patience 12, restore best weights), ReduceLROnPlateau (×0.5 after 5 epochs), ModelCheckpoint, all watching `val_loss`.

---

## Key Findings

1. **86.14% test accuracy from scratch**, with training and validation accuracy finishing within 0.3 points of each other, so there is no overfitting.
2. **Vehicles are easiest:** automobile (F1 0.941), ship (0.927) and truck (0.908).
3. **Dogs get mistaken for cats:** 137 dogs were predicted as cats, the largest single error. Cat has the lowest F1 (0.740).
4. **Frog is the fallback guess:** the model predicted "frog" 1,310 times for 1,000 frogs (97% recall, 74% precision).
5. **Cars and trucks overlap:** 43 cars were called trucks and 29 trucks were called cars.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python, NumPy | Data handling |
| TensorFlow, Keras | CNN model, augmentation, training |
| Scikit-learn | Stratified split, confusion matrix, classification report |
| Matplotlib, Seaborn | Visualisation |
| Gradio | Interactive app inside the notebook |
| Google Colab (T4 GPU) | Training |
| HTML / CSS / JavaScript, Chart.js | Portfolio webpage and in-browser demo |

---

## How the Live Demo Works

The demo runs the trained model (`cnn_cifar10_final.keras`) **directly in the browser** with a small pure-JavaScript inference engine. No TensorFlow.js, no server, and photos are never uploaded.

- Convolution weights are stored as 16-bit floats and dense weights as 8-bit integers, shrinking the model to about 2.2 MB (`model.js`).
- The browser version gives the same prediction as the full Keras model on 999 of 1,000 test images.
- Uploaded photos are centre-cropped to a square and resized to 32×32, the same steps as the notebook's Gradio `predict()` function.

---

## Repository Structure

```
├── index.html                      # Portfolio page with live demo and charts
├── styles.css                      # Page styling, light/dark themes, responsive layout
├── model.js                        # Trained CNN weights for the in-browser demo
├── Image_Classification_CNN.ipynb  # Full training notebook (Colab-ready)
├── cnn_cifar10_final.keras         # Trained Keras model (optional, 15.5 MB)
└── README.md                       # Project documentation
```

---

## Run Locally

**Webpage:** open `index.html` in a modern browser. Keep `model.js` and `styles.css` in the same folder. The demo works offline; the charts and fonts load from a CDN.

**Notebook:** open `Image_Classification_CNN.ipynb` in Google Colab, choose `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. Or run it locally:

```bash
git clone https://github.com/akram23679/CIFAR10-Image-Classification.git
cd CIFAR10-Image-Classification
pip install tensorflow scikit-learn matplotlib seaborn pillow gradio jupyter
jupyter notebook Image_Classification_CNN.ipynb
```

**Use the saved model in Python:**

```python
import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model("cnn_cifar10_final.keras")
img = Image.open("photo.jpg").convert("RGB").resize((32, 32))
probs = model.predict(np.asarray(img, dtype="float32")[None] / 255.0)[0]
```

## Deploy with GitHub Pages

1. Push the files to your repository.
2. Go to **Settings → Pages**.
3. Under **Source**, choose the `main` branch and the `/ (root)` folder.
4. Save. The site will be live at https://akram23679.github.io/CIFAR10-Image-Classification/.

---

## Limitations & Next Steps

- Images are shrunk to 32×32, so fine detail in real photos is lost.
- The model knows only 10 classes and always picks one.
- **Next steps:** transfer learning with a pre-trained network (ResNet, EfficientNet), residual connections, stronger augmentation (Cutout, MixUp) and Grad-CAM heatmaps for explainability.

---

## Author

**Akram Ali**  
Data Analyst  
GitHub: [@akram23679](https://github.com/akram23679)
