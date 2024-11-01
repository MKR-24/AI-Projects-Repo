import streamlit as st
from pathlib import Path
import google.generativeai as genai

from api_key import api_key

genai.configure(api_key=api_key)

generation_config = {
  "temperature": 0.4,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}
safety_settings=[
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]
system_prompts="""
You are a specialized medical image analysis assistant with the following responsibilities:

CORE RESPONSIBILITIES:
1. Image Analysis Protocol:
- Conduct thorough analysis of medical images
- Follow a systematic approach to identify key features
- Document all visible anomalies or pathological findings
- Maintain clinical objectivity in assessments

2. Reporting Structure:
- Provide detailed, organized findings reports
- List observations in clear medical terminology
- Include measurements and comparative analysis where relevant
- Flag critical findings that require immediate attention

3. Clinical Recommendations:
- Suggest evidence-based next steps for investigation
- Outline potential diagnostic pathways
- Provide relevant differential diagnoses
- Recommend appropriate follow-up imaging if needed

OPERATIONAL GUIDELINES:
- Analyze only medically relevant images
- Note image quality limitations that may affect interpretation
- Include appropriate medical disclaimers
- Use standardized medical terminology
- Structure responses in a clear, professional format

LIMITATIONS AND DISCLAIMERS:
- Clearly state that analyses are supportive tools, not definitive diagnoses
- Recommend consultation with healthcare providers for final interpretation
- Acknowledge when image quality or data is insufficient for analysis
- Maintain patient privacy and confidentiality standards

"""
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  safety_settings=safety_settings
)


st.set_page_config(page_title="Medical Image Analytics", page_icon=":robot")
st.image("MedicalLogo.jpg",width=150)
st.markdown("""
    <div style="display: flex; align-items: center; font-size: 32px; gap: 15px;">
        <span>🧑🏻‍⚕️ Medical</span>
        <span>🌐 Image</span>
        <span>🩻 Analytics</span>
        <span>📊 Bot 🤖</span>
    </div>
""", unsafe_allow_html=True)

st.subheader("An application that can help users to identify medical images")
uploaded_file=st.file_uploader("Upload image for analysis",type=["png","jpg","jpeg"])
if uploaded_file:
    st.image(uploaded_file,width=300, caption="Uploaded Medical Image")
submit_btn=st.button("Generate the Result")

if submit_btn:
    image_data=uploaded_file.getvalue()
    
    image_parts=[
        {
            "mime_type":"image/jpeg",
            "data":image_data
        },
    ]
    prompt_parts=[
        image_parts[0],
        system_prompts,
    ]
    st.title("Analysis Results:")
    response=model.generate_content(prompt_parts)
    st.write(response.text)