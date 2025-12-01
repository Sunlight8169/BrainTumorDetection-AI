import streamlit as st  # type: ignore
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chatbot_engine import ChatbotEngine
import requests
from PIL import Image
import io
from reportlab.pdfgen import canvas # pyright: ignore[reportMissingModuleSource]
from reportlab.lib.pagesizes import A4 # pyright: ignore[reportMissingModuleSource]
from reportlab.lib.colors import HexColor # pyright: ignore[reportMissingModuleSource]
from reportlab.lib.units import mm # pyright: ignore[reportMissingModuleSource]
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Medical Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000 !important;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: 15%;
        border-left: 4px solid #2196F3;
        color: #000 !important;
    }
    .bot-message {
        background-color: #F5F5F5;
        margin-right: 15%;
        border-left: 4px solid #4CAF50;
        color: #000 !important;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .high-confidence {
        background-color: #4CAF50;
        color: white;
    }
    .medium-confidence {
        background-color: #FF9800;
        color: white;
    }
    .low-confidence {
        background-color: #F44336;
        color: white;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFC107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- PDF HELPER FUNCTION (UPDATED FORMAT) ---------------- #

def generate_pdf_report(patient_name, patient_age, patient_gender, tumor_type_raw, confidence):
    """
    Generate a professional PDF report for Brain MRI analysis.
    Returns an in-memory BytesIO buffer containing the PDF.
    """

    # Safe defaults if fields are empty
    patient_name = patient_name.strip() if patient_name else "Not Provided"
    patient_age = patient_age.strip() if patient_age else "Not Provided"
    patient_gender = patient_gender.strip() if patient_gender else "Not Provided"

    # Map raw prediction to human-readable info
    tumor_details = {
        "glioma_tumor": {
            "display_name": "Glioma Tumor",
            "location": "Glioma usually occurs in the brain or spinal cord and arises from glial cells.",
            "severity": "High – considered dangerous and may be malignant.",
            "description": (
                "Glioma is a serious type of brain tumor that can grow aggressively and may affect "
                "brain function, causing headaches, seizures, personality changes, and other neurological symptoms. "
                "Early medical evaluation and treatment are very important."
            )
        },
        "meningioma_tumor": {
            "display_name": "Meningioma Tumor",
            "location": "Meningioma arises from the meninges – the protective membranes covering the brain and spinal cord.",
            "severity": "Moderate – often slow-growing and sometimes operable.",
            "description": (
                "Meningioma is usually a slow-growing tumor that may remain silent for a long time. "
                "It can cause headaches, weakness, or vision problems depending on its size and location. "
                "Many meningiomas are benign but still need medical supervision."
            )
        },
        "pituitary_tumor": {
            "display_name": "Pituitary Tumor",
            "location": "Pituitary tumor is found in the pituitary gland at the base of the brain.",
            "severity": "Low to Moderate – often non-cancerous but can affect hormones.",
            "description": (
                "Pituitary tumors are commonly benign but can disturb hormone levels, leading to fatigue, "
                "weight changes, vision problems, or other hormonal symptoms. "
                "Treatment depends on size, type, and hormone activity."
            )
        },
        "no_tumor": {
            "display_name": "No Tumor Detected",
            "location": "No tumor region identified in the visible brain MRI scan.",
            "severity": "None – no tumor detected by the AI model.",
            "description": (
                "The AI model did not detect any tumor-like abnormality in this MRI scan. "
                "However, if symptoms persist, you should still consult a qualified doctor for a full evaluation."
            )
        }
    }

    info = tumor_details.get(
        tumor_type_raw,
        {
            "display_name": tumor_type_raw.replace("_", " ").title() if tumor_type_raw else "Unknown",
            "location": "Information not available.",
            "severity": "Unknown",
            "description": "No detailed description available for this condition."
        }
    )

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin = 20 * mm
    y = height - margin

    # ====== HEADER BAR (Blue, Hospital Style) ======
    header_h = 22 * mm
    c.setFillColor(HexColor("#0059B3"))
    c.rect(0, height - header_h, width, header_h, fill=1, stroke=0)

    # Hospital Name
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 10 * mm, "Apollo Brain & Spine Scan Centre")

    # Report Title
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, height - 17 * mm, "Brain MRI Analysis Report (AI-Assisted)")

    # Reset color to black for body
    c.setFillColor(HexColor("#000000"))

    # ====== REPORT META (Date & Time) ======
    y = height - header_h - 12 * mm  # INCREASED GAP
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Report Date & Time : {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    y -= 8
    c.setLineWidth(0.5)
    c.setStrokeColor(HexColor("#999999"))
    c.line(margin, y, width - margin, y)
    y -= 18  # INCREASED GAP after line

    # ====== PATIENT INFORMATION BOX ======
    box_h = 42 * mm  # INCREASED BOX HEIGHT
    c.setFillColor(HexColor("#E7F0FA"))
    c.roundRect(margin, y - box_h + 5 * mm, width - 2 * margin, box_h, 5, fill=1, stroke=0)

    c.setFillColor(HexColor("#0059B3"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin + 5 * mm, y, "Patient Information")

    c.setFillColor(HexColor("#000000"))
    c.setFont("Helvetica", 11)
    y -= 16  # INCREASED SPACING
    c.drawString(margin + 7 * mm, y, f"Name   : {patient_name}")
    y -= 16  # INCREASED SPACING
    c.drawString(margin + 7 * mm, y, f"Age    : {patient_age}")
    y -= 16  # INCREASED SPACING
    c.drawString(margin + 7 * mm, y, f"Gender : {patient_gender}")

    y -= 24  # INCREASED GAP after box

    # ====== AI ANALYSIS RESULT SECTION ======
    c.setFillColor(HexColor("#0059B3"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "AI Analysis Result")
    y -= 16  # INCREASED SPACING
    c.setFillColor(HexColor("#000000"))
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Predicted Tumor Type : {info['display_name']}")
    y -= 16  # INCREASED SPACING
    c.drawString(margin, y, f"Model Confidence     : {confidence:.2f}%")
    y -= 16  # INCREASED SPACING
    c.drawString(margin, y, f"Severity Level       : {info['severity']}")

    y -= 24  # INCREASED GAP before next section

    # ====== ABOUT CONDITION BOX ======
    c.setFillColor(HexColor("#E7F0FA"))
    about_box_start_y = y
    about_box_h = 75 * mm  # INCREASED BOX HEIGHT
    c.roundRect(margin, about_box_start_y - about_box_h + 5 * mm, width - 2 * margin, about_box_h, 5, fill=1, stroke=0)

    c.setFillColor(HexColor("#0059B3"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin + 5 * mm, y, "About The Detected Condition")

    y -= 18  # INCREASED SPACING
    c.setFillColor(HexColor("#000000"))
    text = c.beginText()
    text.setTextOrigin(margin + 7 * mm, y)
    text.setFont("Helvetica", 10)
    text.setLeading(14)  # INCREASED LINE SPACING
    
    # Wrap location text to fit properly
    location_text = f"Typical Location: {info['location']}"
    max_chars = 100
    location_line = ""
    for word in location_text.split():
        if len(location_line) + len(word) + 1 <= max_chars:
            location_line += (word + " ")
        else:
            text.textLine(location_line.strip())
            location_line = word + " "
    if location_line:
        text.textLine(location_line.strip())
    
    text.textLine("")  # Empty line for spacing

    # Wrap description text nicely
    desc = info["description"]
    max_chars = 100
    line = ""
    for word in desc.split():
        if len(line) + len(word) + 1 <= max_chars:
            line += (word + " ")
        else:
            text.textLine(line.strip())
            line = word + " "
    if line:
        text.textLine(line.strip())

    c.drawText(text)
    y = text.getY() - 18  # INCREASED GAP

    # ====== REPORT GENERATED BY ======
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Report Generated By:")
    y -= 16  # INCREASED SPACING
    c.setFont("Helvetica", 10)
    c.drawString(margin + 7 * mm, y, "• Suraj Vishwakarma")
    y -= 16  # INCREASED SPACING
    c.drawString(margin + 7 * mm, y, "• Monu Kumar Jha")
    y -= 18  # INCREASED SPACING
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin + 7 * mm, y, "Academic Project: Brain Tumor Detection & Medical AI Chatbot")

    # ====== FOOTER BAR WITH DISCLAIMER ======
    footer_h = 28 * mm  # INCREASED FOOTER HEIGHT
    c.setFillColor(HexColor("#0059B3"))
    c.rect(0, 0, width, footer_h, fill=1, stroke=0)

    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(width / 2, 18 * mm, "Disclaimer")

    c.setFont("Helvetica", 8)
    c.drawCentredString(width / 2, 13 * mm,
                        "This report is generated only for educational and research purposes.")
    c.drawCentredString(width / 2, 8 * mm,
                        "It must NOT be used for real medical diagnosis, treatment, or clinical decisions.")
    c.drawCentredString(width / 2, 4 * mm,
                        "Always consult a qualified doctor.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------- CHATBOT INIT ---------------- #

@st.cache_resource
def load_chatbot():
    return ChatbotEngine(similarity_threshold=0.4)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chatbot_loaded' not in st.session_state:
    with st.spinner("🔄 Loading AI models... This may take a minute on first run."):
        st.session_state.chatbot = load_chatbot()
        st.session_state.chatbot_loaded = True

chatbot = st.session_state.chatbot

# Header
st.markdown('<h1 class="main-header">🧠 Brain Tumor Medical Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Medical Chatbot & MRI Analysis System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About This System")
    st.write("""
    This AI-powered system provides:
    
    🤖 **Intelligent Chatbot**
    - Brain tumor information
    - Symptoms & diagnosis
    - Treatment options
    - Medical guidance
    
    🔬 **MRI Analysis**
    - AI-powered tumor detection
    - Classification (Glioma, Meningioma, Pituitary)
    - Instant results
    """)
    
    st.divider()
    
    st.header("🎯 Quick Questions")
    quick_questions = [
        "What is a brain tumor?",
        "What are the symptoms?",
        "What is glioma?",
        "What is meningioma?",
        "What is a pituitary tumor?",
        "How are tumors diagnosed?",
        "What are treatment options?",
        "Can I upload my MRI?"
    ]
    
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            response = chatbot.get_response(q)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
        st.rerun()
    
    st.divider()
    st.caption("⚕️ Medical AI Project")
    st.caption("Powered by Deep Learning & NLP")

# Main content area with tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📷 MRI Analysis"])

# TAB 1: Chat Assistant
with tab1:
    st.subheader("💬 Ask Questions About Brain Tumors")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Hi! I'm your Brain Tumor Medical Assistant. Ask me anything about brain tumors, or use the quick questions in the sidebar!")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                response = message["content"]
                confidence = response.get('confidence', 0)
                
                # Determine confidence level
                if confidence >= 70:
                    conf_class = "high-confidence"
                    conf_text = "High"
                elif confidence >= 40:
                    conf_class = "medium-confidence"
                    conf_text = "Medium"
                else:
                    conf_class = "low-confidence"
                    conf_text = "Low"
                
                st.markdown(f'''
                <div class="chat-message bot-message">
                    🤖 <strong>Assistant:</strong><br>{response["answer"]}
                    <br>
                    <span class="confidence-badge {conf_class}">🎯 Confidence: {confidence}% ({conf_text})</span>
                </div>
                ''', unsafe_allow_html=True)
    
    # User input
    st.divider()
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "💭 Type your question here...",
            key="user_input",
            placeholder="e.g., What are the symptoms of glioma?",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("Send ➤", use_container_width=True, type="primary")
    
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            response = chatbot.get_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()

# TAB 2: MRI Analysis
with tab2:
    st.subheader("📷 Upload Brain MRI for AI Analysis")
    
    # Warning message
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Important:</strong> This is an AI screening tool for educational purposes. 
        Results should ALWAYS be verified by qualified medical professionals and radiologists.
    </div>
    """, unsafe_allow_html=True)

    # Patient info inputs
    st.markdown("### 🧾 Patient Information (Optional but Recommended)")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        patient_name = st.text_input("Patient Name", value="", placeholder="e.g., Rahul Kumar")
    with p_col2:
        patient_age = st.text_input("Age", value="", placeholder="e.g., 35")
    with p_col3:
        patient_gender = st.text_input("Gender", value="", placeholder="e.g., Male / Female / Other")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an MRI scan image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan in JPG or PNG format"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="📁 Uploaded MRI Scan", use_container_width=True)
    
    with col2:
        if uploaded_file:
            if st.button("🔍 Analyze MRI Scan", use_container_width=True, type="primary"):
                with st.spinner("🧠 Analyzing brain MRI... Please wait..."):
                    try:
                        # Prepare image
                        image = Image.open(uploaded_file)
                        
                        # Convert to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        
                        # Send to Flask backend
                        files = {'file': ('image.png', img_byte_arr, 'image/png')}
                        response = requests.post('http://localhost:5000/predict', files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            prediction = result.get('prediction', 'Unknown')
                            confidence = result.get('confidence', 0)
                            
                            # Display result
                            st.success("✅ Analysis Complete!")
                            
                            # Prediction result
                            st.metric(label="🎯 Detected Condition", value=prediction.upper())
                            st.metric(label="📊 Confidence Level", value=f"{confidence}%")
                            
                            # All predictions (if available)
                            if 'all_predictions' in result:
                                st.subheader("📈 Detailed Probabilities")
                                all_preds = result['all_predictions']
                                for tumor_type, prob in sorted(all_preds.items(), key=lambda x: x[1], reverse=True):
                                    st.progress(prob / 100, text=f"{tumor_type.upper()}: {prob:.2f}%")
                            
                            # PDF report generation
                            pdf_buffer = generate_pdf_report(
                                patient_name=patient_name,
                                patient_age=patient_age,
                                patient_gender=patient_gender,
                                tumor_type_raw=prediction,
                                confidence=confidence
                            )

                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer,
                                file_name="Brain_MRI_Report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            # Add to chat history
                            bot_response = {
                                'answer': (
                                    f"🔬 **MRI Analysis Complete**\n\n"
                                    f"**Detected:** {prediction.upper()}\n"
                                    f"**Confidence:** {confidence}%\n\n"
                                    f"⚕️ Please consult a radiologist or neurologist for professional interpretation "
                                    f"and treatment planning."
                                ),
                                'confidence': confidence,
                                'category': 'mri_analysis',
                                'matched': True
                            }
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": bot_response
                            })
                            
                            st.info("💬 Result added to chat history!")
                        
                        else:
                            st.error(f"❌ Error: Server returned status code {response.status_code}")
                            st.error(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ **Cannot connect to prediction server!**")
                        st.warning("""
                        **Please make sure:**
                        1. Flask backend is running: `python backend/main.py`
                        2. Server is accessible at `http://localhost:5000`
                        3. Check firewall settings
                        """)
                    
                    except Exception as e:
                        st.error(f"❌ **Error during analysis:** {str(e)}")
                        st.info("Check console for detailed error information")
        else:
            st.info("👆 Upload an MRI scan to begin analysis")
            
            # Sample info
            with st.expander("ℹ️ What images can I upload?"):
                st.write("""
                **Accepted formats:** JPG, JPEG, PNG
                
                **Best practices:**
                - Use clear, high-quality MRI scans
                - Brain should be clearly visible
                - Preferred: T1 or T2-weighted MRI images
                - Avoid blurry or low-resolution images
                
                **Detectable conditions:**
                - Glioma
                - Meningioma
                - Pituitary Tumor
                - No Tumor (Normal)
                """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='font-size: 0.9rem;'>
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only.<br>
        Always seek professional medical advice for diagnosis, treatment, and medical decisions.
    </p>
    <p style='font-size: 0.8rem; color: #999;'>
        🎓 Major Project | Brain Tumor Detection & Medical AI Chatbot<br>
        Powered by TensorFlow, Sentence Transformers & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
