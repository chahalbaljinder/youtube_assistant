import streamlit as st
import langchain_helper as lch
import textwrap

# Inject Custom CSS
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #1E1E1E;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2C2C2C;
        color: white;
    }
    .stTextInput, .stTextArea {
        background-color: #333333 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🎥 YouTube Assistant 🤖")

# Sidebar Inputs
with st.sidebar:
    st.header("📌 Video & Query Input")
    google_api_key = st.text_input("🔑 Enter Google API Key:", type="password", max_chars=100)
    youtube_url = st.text_input("🎬 Paste YouTube URL:", max_chars=100)
    query = st.text_area("❓ Ask a question about the video:", max_chars=200)
    submit_button = st.button("🚀 Get Answer")

# Process Query
if submit_button:
    if not google_api_key:
        st.error("⚠️ Please enter a Google API Key.")
    elif not youtube_url:
        st.error("⚠️ Please enter a YouTube URL.")
    elif not query:
        st.error("⚠️ Please enter a query.")
    else:
        with st.spinner("🔍 Processing... Please wait."):
            try:
                db = lch.create_db_from_youtube_video_url(youtube_url, google_api_key)
                response, docs = lch.get_response_from_query(db, query, google_api_key)
                st.success("✅ Answer Generated!")
                st.subheader("📜 Answer:")
                st.write(textwrap.fill(response, width=180))

                with st.expander("📑 See Related Context:"):
                    for doc in docs:
                        st.text(textwrap.fill(doc.page_content, width=180))
            except Exception as e:
                st.error(f"🚨 Error: {str(e)}")
else:
    st.info("⚠️ Please enter a Google API Key, YouTube URL, and query before submitting.")