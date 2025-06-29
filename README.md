
# Audio and Visual Features-Based Cascade for Multimedia Content Matching

This project presents an efficient and accurate methodology for detecting multimedia content—such as **advertisements**—in reference store boradcast video. The approach employs a **cascade of heterogeneous models** that combine fast audio-based detection with precise visual similarity verification.


## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/muhammadali-msc/Audio-and-Visual-Features-Based-Cascade-For-Multimedia-Content-Matching.git
cd Audio-and-Visual-Features-Based-Cascade-For-Multimedia-Content-Matching
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

A file explorer will open where you can select:
- 📁 A **reference** video (e.g., full-length broadcast)
- 📁 A **query** video (e.g., ad or clip to find)
- 🧠 Select Detection Method
    1. Visual Detection Model
    2. Audio Detection Model
    3. Cascade Multimedia Detection Model

## 📦 Dependencies

Make sure the following libraries are installed:

```
moviepy
librosa
opencv-python
pandas
numpy
scikit-image
plotly
image-similarity-measures
tkinter (included with Python, or install manually on Linux)
```

## 🛡️ .gitignore

Ensure the `.gitignore` file excludes:

```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.env
```

## 📊 Example Use Case

This system has been tested on multi-hour broadcast recordings to identify:
- 📺 Advertisement occurrences
- 📆 Airtime tracking
- 🎯 Duplicate content across archives