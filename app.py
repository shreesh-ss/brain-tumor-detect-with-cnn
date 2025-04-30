import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
from PIL import Image
# from appointment import book_appointment
from tensorflow.keras.models import load_model


def preprocess_image(image):
    # Resize to match input size
    img = image.resize((128, 128))
    # Convert image to numpy array and normalize
    img = np.array(img) / 255.0
    # Ensure the image has 3 channels (convert if grayscale)
    if len(img.shape) == 2:  # Grayscale image
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] != 3:  # Other number of channels
        raise ValueError("Unexpected number of channels in image")
    # Expand dimensions to match model's input shape
    img = np.expand_dims(img, axis=0)
    return img


def book_appointment():
    st.subheader("📞 Book an Appointment with a Specialist")

    # Doctor information
    doctors = [
        {"name": "Dr. John Doe", "specialization": "Neurosurgeon",
            "contact": "+1 555-123-4567", "email": "doctor1@example.com"},
        {"name": "Dr. Jane Smith", "specialization": "Neurologist",
            "contact": "+1 555-987-6543", "email": "doctor2@example.com"},
        {"name": "Dr. Robert Brown", "specialization": "Radiologist",
            "contact": "+1 555-456-7890", "email": "doctor3@example.com"}
    ]

    # Display doctor information
    doctor_options = [doctor["name"] for doctor in doctors]
    selected_doctor = st.selectbox("👨‍⚕️ Select a doctor", doctor_options)

    # Define available time slots
    time_slots = ["10:00 AM", "11:00 AM", "3:00 PM", "4:00 PM", "5:00 PM", "7:00 PM"]

    # Appointment booking form
    with st.form(key="appointment_form"):
        name = st.text_input("👤 Your Name")
        email = st.text_input("📧 Your Email Address")
        contact = st.text_input("📞 Your Contact Number")
        city = st.text_input("🏙️ City")
        state = st.text_input("🌆 State")
        country = st.text_input("🌍 Country")
        date = st.date_input("📅 Preferred Appointment Date")
        selected_time_slot = st.selectbox("⏰ Select a time slot", time_slots)
        message = st.text_area("💬 Message (optional)")

        # Submit button
        submit_button = st.form_submit_button("📤 Submit Appointment Request")

        if submit_button:
            # Find selected doctor's details
            doctor = next(doctor for doctor in doctors if doctor["name"] == selected_doctor)
            doctor_email = doctor["email"]

            subject = "🩺 New Appointment Request"
            body_to_doctor = f"""
            Appointment Request from {name} ({email}):
            🗓️ Appointment Date: {date}
            Time Slot: {selected_time_slot}
            Message: {message}
            """
            # Send email to doctor
            send_email(doctor_email, subject, body_to_doctor)

            st.success(f"✅ Appointment request sent! You will receive a confirmation email shortly. 📧")

uploaded_file = './mental_health_diagnosis_treatment_.csv'

data = pd.read_csv(uploaded_file)
st.set_page_config(page_title="Brain Diagnosis & Appointment", page_icon="🩺", layout="wide")
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .main {
            padding: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)
# Streamlit app setup
st.title("Brain Tumor Analysis using CNN")

with st.sidebar:
    menu = option_menu('Mental Health Diagnosis and Treatment Analysis',
                              ['Tumor detection','📅 Book an Appointment'],
                              icons=['dashboard','activity','heart','person','line-chart'],
                              default_index=0)

if menu == "Overview":
    st.header("Dataset Overview")
    st.write("Here are the first few rows of the dataset:")
    st.dataframe(data.head(20))

elif menu == "Statistics":
    st.header("Descriptive Statistics")
    st.write("The following table shows key statistical measures:")
    st.write(data.describe())


elif menu == "Visualizations":
    st.header("Data Visualizations")

    # Additional Visualizations
    st.subheader("Distributions of Key Columns")
    columns_to_plot = [ 'Age', 'Symptom Severity (1-10)', 'Mood Score (1-10)', 'Sleep Quality (1-10)',
                       'Physical Activity (hrs/week)', 'Treatment Duration (weeks)', 'Stress Level (1-10)',
                       'Treatment Progress (1-10)', 'Adherence to Treatment (%)']

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(3, 4, i)
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')

    plt.tight_layout()
    st.pyplot(plt)

    # Select a column for visualization
    columns = data.columns[1:]
    selected_column = st.selectbox("Select a column to visualize", columns)

    if data[selected_column].dtype in ['int64', 'float64']:
        st.subheader(f"Distribution of {selected_column}")
        fig, ax = plt.subplots()
        sns.histplot(data[selected_column], kde=True, ax=ax)
        st.pyplot(fig)

    elif data[selected_column].dtype == 'object':
        st.subheader(f"Counts of {selected_column}")
        fig, ax = plt.subplots()
        data[selected_column].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    else:
        st.write("Visualization for this data type is not supported.")

elif menu == "Tumor detection":
    model = load_model("brain_tumor_cnn_model.h5")
    st.title("🧠 Brain Tumor Detection ")
    
    st.subheader(
            "🔬 Upload an MRI image to check for the presence of a brain tumor.")
    uploaded_file = st.file_uploader(
        "Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Preprocess the image and make prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Display the result with emojis for visual feedback
        if predicted_class == 1:
            st.error(
                "⚠️ Tumor detected! Please consult a healthcare provider immediately. 🏥")
        else:
            st.success(
                "✅ No tumor detected. Keep up with regular health check-ups to stay healthy! 💪")

elif menu == "📅 Book an Appointment":
    book_appointment()
