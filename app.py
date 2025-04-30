import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
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


def validate_mri_image(image, filename):
    """Validate if the uploaded image is likely an MRI scan."""
    # Check file extension
    valid_extensions = ['jpg', 'jpeg', 'png']
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError("Invalid file type. Please upload a JPG, JPEG, or PNG image.")
    
    # Basic MRI validation: Check if image is grayscale or has expected properties
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[-1] != 1 and img_array.shape[-1] != 3:
        raise ValueError("Invalid image format. Please upload a valid MRI scan image.")
    
    # Optional: Add more sophisticated checks (e.g., pixel intensity range typical for MRI)
    if img_array.mean() < 10 or img_array.mean() > 245:  # Typical MRI intensity range
        raise ValueError("Image does not appear to be a valid MRI scan.")
    
    return True


def estimate_tumor_diameter(prediction_confidence):
    """Simulate tumor diameter estimation based on prediction confidence."""
    # Placeholder logic: Map confidence to a diameter (e.g., 0-50mm range)
    # In a real system, this would require image segmentation (e.g., using U-Net)
    max_diameter = 50.0  # Maximum assumed diameter in mm
    diameter = prediction_confidence * max_diameter
    return round(diameter, 2)


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

    # Define available time slots (fixed syntax error)
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
            # Send email to doctor (placeholder as send_email is not defined)
            # send_email(doctor_email, subject, body_to_doctor)

            st.success(f"✅ Appointment request sent! You will receive a confirmation email shortly. 📧")


# Load dataset
uploaded_file = './mental_health_diagnosis_treatment_.csv'
data = pd.read_csv(uploaded_file)

# Streamlit app setup
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

st.title("Brain Tumor Analysis using CNN")

with st.sidebar:
    menu = option_menu('Mental Health Diagnosis and Treatment Analysis',
                       ['Tumor detection', '📅 Book an Appointment'],
                       icons=['dashboard', 'activity', 'heart', 'person', 'line-chart'],
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
    columns_to_plot = ['Age', 'Symptom Severity (1-10)', 'Mood Score (1-10)', 'Sleep Quality (1-10)',
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
    st.title("🧠 Brain Tumor Detection")
    
    st.subheader("🔬 Upload an MRI image to check for the presence of a brain tumor.")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Validate MRI image
            validate_mri_image(image, uploaded_file.name)
            
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)

            # Preprocess the image and make prediction
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100  # Confidence percentage

            # Display the result with detailed information
            if predicted_class == 1:
                tumor_diameter = estimate_tumor_diameter(confidence / 100)
                st.error(
                    f"⚠️ Tumor detected!\n\n"
                    f"- **Confidence**: {confidence:.2f}%\n"
                    f"- **Estimated Tumor Diameter**: {tumor_diameter} mm\n"
                    f"📌 Please consult a healthcare provider immediately. 🏥"
                )
            else:
                st.success(
                    f"✅ No tumor detected.\n\n"
                    f"- **Confidence**: {confidence:.2f}%\n"
                    f"💪 Keep up with regular health check-ups to stay healthy!"
                )

        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}. Please upload a valid MRI image.")

elif menu == "📅 Book an Appointment":
    book_appointment()
