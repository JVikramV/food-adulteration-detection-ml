# 🧪 Food Adulteration Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/PyTorch-red)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end machine learning system to detect food adulteration in **chilli powder (image-based)** and **milk (spectral data-based)**. The project provides real-time predictions, interpretability via heatmaps, and an interactive web interface.

---

## 🚀 Features

### 🌶️ Chilli Powder Analysis

* Predicts **percentage of adulteration** from input images
* Generates **Grad-CAM heatmaps** to highlight adulterated regions
* Classifies samples as:

  * ✅ Pure
  * ⚠️ Adulterated

### 🥛 Milk Analysis

* Uses **spectral data (CSV input)**
* Detects:

  * Presence of adulteration
  * Type of adulterant
* Displays **spectral curves for analysis**

---

## 🧠 Tech Stack

* **Machine Learning:** PyTorch, Scikit-learn
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, OpenCV
* **Frontend/UI:** Gradio
* **Backend:** Python

---

## 📂 Project Structure

```
food-adulteration-detection-ml/
│── src/
│   ├── train.py
│   ├── predict.py
│   ├── eval.py
│   ├── demo_gradio.py
│   ├── models.py
│   ├── gradcam.py
│   ├── dataset_image.py
│   ├── dataset_spectra.py
│   ├── preprocessing/
│── README.md
│── requirements.txt
│── .gitignore
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/JVikramV/food-adulteration-detection-ml.git
cd food-adulteration-detection-ml
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### 🔥 Launch Web Interface (Recommended)

```
python src/demo_gradio.py
```

Open in browser:

```
http://127.0.0.1:7860
```

---

### 🔹 Train Model

```
python src/train.py
```

---

### 🔹 Run Predictions

```
python src/predict.py
```

---

## 📦 Model Weights & Dataset

Due to GitHub size limitations, trained models and datasets are hosted externally.

### 🔗 Download Links

* Chilli Model: https://drive.google.com/file/d/15aQU3pUtpLCkPs6moYdFQXFvnWc6UO6a/view?usp=sharing
* Milk Model: https://drive.google.com/file/d/1pn_ZIZlcfPK0XultwxXZ05es6c_LYz8m/view?usp=sharing


---

### 📌 After Download

Place the files in the following structure:

```
project-root/
│── models/
│   ├── chilli_model.pth
│   ├── milk_model.pth
```

---

### ⚠️ Note

If models are not downloaded, you can train them using:

```
python src/train.py
```

---

## 🐍 Python Version

This project is tested on:

* Python **3.10**

⚠️ PyTorch may not work correctly with Python 3.13+

---

## 📊 Output

### Chilli Model

* Adulteration percentage
* Heatmap visualization (Grad-CAM)
* Classification (Pure / Adulterated)

### Milk Model

* Adulterant detection
* Spectral curve visualization

---

## 📸 Demo

<img width="1918" height="825" alt="Screenshot 2026-03-17 192822" src="https://github.com/user-attachments/assets/c14dc1a0-ab52-489d-979f-b16acd853053" />
<img width="1847" height="791" alt="Screenshot 2026-03-17 192716" src="https://github.com/user-attachments/assets/d7b54e53-cdc4-4036-82c3-d67cbe27984c" />



```
images/
│── heatmap.png
│── ui.png
│── spectral.png
```

```
![Heatmap](images/heatmap.png)
![UI](images/ui.png)
![Spectral](images/spectral.png)
```

---

## 🏆 Achievements

* 🥈 **Runner-Up at Tech Summit**
* Developed a solution addressing **real-world food safety challenges**

---

## 🌍 Impact

* Enhances **food quality monitoring**
* Supports **consumer safety**
* Demonstrates practical ML application in **real-world scenarios**

---

## 🔮 Future Improvements

* Mobile application integration
* Real-time camera-based detection
* Expansion to more food categories
* Cloud deployment (AWS / GCP)

---

## 👨‍💻 Author

**Jayanth Vikram**

* Computer Science Engineering Student
* Interested in AI,ML, Backend Development, and Scalable Systems

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
