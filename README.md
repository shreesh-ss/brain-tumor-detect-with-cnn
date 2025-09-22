# Brain Tumor Detection with CNN

A deep learning project for automated brain tumor detection from MRI images using a Convolutional Neural Network (CNN). This project also provides a Streamlit web app for easy image-based diagnosis and appointment booking with specialists. Additionally, it includes mental health diagnosis and treatment analysis from a provided dataset.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture & Training](#model-architecture--training)
- [How the Model Works](#how-the-model-works)
- [Dataset](#dataset)
- [Web App Functionality](#web-app-functionality)
- [Appointment Booking](#appointment-booking)
- [Mental Health Analysis](#mental-health-analysis)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
This project leverages deep learning to detect brain tumors from MRI images. It features:
- A trained CNN model for binary classification (tumor/no tumor)
- A Streamlit web app for user-friendly diagnosis and appointment booking
- Data analysis and visualization for mental health diagnosis and treatment outcomes

## Features
- **Brain Tumor Detection**: Upload an MRI image and get instant predictions.
- **Appointment Booking**: Book a consultation with a specialist directly from the app.
- **Mental Health Data Analysis**: Visualize and analyze mental health diagnosis and treatment data.

## Demo
To see the app in action, run the following command and open the provided local URL:
```bash
streamlit run app.py
```

---

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shreesh-ss/brain-tumor-detect-with-cnn.git
   cd brain-tumor-detect-with-cnn
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dataset:**
   - Place MRI images in `brain_tumor_dataset/yes` and `brain_tumor_dataset/no`.
   - The mental health dataset should be named `mental_health_diagnosis_treatment_.csv`.
4. **(Optional) Train the model:**
   - Use the provided notebook `brain-tumor-detect-with-cnn.ipynb` to retrain the model.
   - The trained model is saved as `brain_tumor_cnn_model.h5`.

---

## Usage
- **Run the Streamlit app:**
  ```bash
  streamlit run app.py
  ```
- **Web App Sections:**
  - **Tumor Detection**: Upload an MRI image to get a prediction.
  - **Book an Appointment**: Fill out a form to book a specialist consultation.
  - **Mental Health Analysis**: Explore and visualize mental health data.

---

## Model Architecture & Training
- **Model**: Convolutional Neural Network (CNN) built with TensorFlow/Keras
- **Input**: MRI images resized to 128x128 pixels, normalized
- **Layers**:
  - Multiple Conv2D and MaxPooling2D layers
  - Dense layers for classification
- **Output**: Softmax layer for binary classification (tumor/no tumor)
- **Training**:
  - Data loaded from `brain_tumor_dataset/yes` and `brain_tumor_dataset/no`
  - Images are labeled (1 for tumor, 0 for no tumor)
  - Model trained for up to 20 epochs with early stopping at 99% accuracy
  - Model saved as `brain_tumor_cnn_model.h5`

---

## How the Model Works
1. **Image Preprocessing**:
   - Uploaded MRI images are resized to 128x128 pixels
   - Images are normalized and converted to 3 channels if needed
2. **Prediction**:
   - The preprocessed image is passed to the trained CNN model
   - The model outputs a probability for each class (tumor/no tumor)
   - The class with the highest probability is selected as the prediction
3. **Result Display**:
   - If a tumor is detected, a warning is shown
   - If no tumor is detected, a success message is displayed

---

## Dataset
- **MRI Images**: Place images in `brain_tumor_dataset/yes` (tumor) and `brain_tumor_dataset/no` (no tumor)
- **Mental Health Data**: `mental_health_diagnosis_treatment_.csv` for analysis and visualization

---

## Web App Functionality
- **Tumor Detection**: Upload an MRI image and get a prediction using the trained model
- **Appointment Booking**: Fill out a form to book a consultation with a specialist
- **Mental Health Analysis**: Visualize and analyze treatment outcomes and statistics

---

## Appointment Booking
- Select a doctor and preferred time slot
- Enter your details and submit the form
- The doctor will receive your request and contact you for confirmation

---

## Mental Health Analysis
- Explore statistics and visualizations from the mental health dataset
- Analyze treatment outcomes, symptom severity, mood scores, and more

---

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License
This project is licensed under the MIT License.
