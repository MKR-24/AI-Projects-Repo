import streamlit as st
import pickle
import re
from io import StringIO
import PyPDF2  # For reading PDFs
import docx  # For reading Word documents


clf = pickle.load(open('clf.pkl', 'rb'))
tfid = pickle.load(open('tfid.pkl', 'rb'))


category_mapping = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and Fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing'
}


def clean_resume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)  # Remove URLs
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)  # Remove hashtags
    cleanText = re.sub(r'@\S+', ' ', cleanText)  # Remove mentions
    cleanText = re.sub(r'RT|cc', ' ', cleanText)  # Remove RT and cc
    cleanText = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""")), ' ', cleanText)  # Remove special characters
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Remove non-ASCII characters
    cleanText = re.sub(r'\s+', ' ', cleanText)  # Replace multiple spaces with a single space
    return cleanText.strip()  # Return the cleaned text


def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    elif file.type == "text/plain":
        return file.read().decode('utf-8')
    else:
        return None


def main():
    st.title("MKR Resume Screening App")
    st.subheader("Upload a resume to predict the professional category.")

    upload_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])

    

    if upload_file is not None:
        
       

        resume_text = extract_text(upload_file)
        if resume_text:

            st.subheader("Uploaded File Preview:")
            st.text_area("File Content", resume_text[:1000], height=300)
            
            cleaned_resume = clean_resume(resume_text)
            cleaned_resume = tfid.transform([cleaned_resume])

         
            prediction_id = clf.predict(cleaned_resume)[0]
            category_name = category_mapping.get(prediction_id, "Unknown Profession")

            
            st.success(f"The predicted category for this resume is: **{category_name}**")
        else:
            st.error("Unsupported file format or unable to read the file. Please upload a valid PDF, DOCX, or TXT file.")

if __name__ == "__main__":
    main()
