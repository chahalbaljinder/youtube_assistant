from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
import google.generativeai as genai
from typing import Any, List, Optional, Dict
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("🚨 GOOGLE_API_KEY is missing. Please check your .env file.")

class GeminiLLM(LLM, BaseModel):
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    _model: Any = None
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        if self._model is None:
            self._model = genai.GenerativeModel(self.model_name)
        
        response = self._model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            )
        )
        
        if response and hasattr(response, "text"):
            return response.text.strip()
        return "⚠️ No response generated. Please try again."

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature
        }

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    """
    Fetches transcript from YouTube, processes it, and creates a FAISS vector store.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()

        if not transcript:
            raise ValueError("⚠️ No transcript found for this video. Please try another one.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        if not docs:
            raise ValueError("⚠️ Failed to split transcript into valid documents.")

        # Ensure embeddings are generated before FAISS indexing
        if embeddings and docs:
            db = FAISS.from_documents(docs, embeddings)
            return db
        else:
            raise ValueError("⚠️ Failed to generate embeddings.")
    
    except Exception as e:
        raise RuntimeError(f"🚨 Error while creating FAISS index: {str(e)}")

def get_response_from_query(db, query, k=4):
    """
    Retrieves the most relevant documents from FAISS and generates a response using Gemini.
    """
    try:
        docs = db.similarity_search(query, k=k)

        if not docs:
            return "⚠️ No relevant information found.", []

        docs_page_content = " ".join([d.page_content for d in docs])

        # Initialize Gemini LLM
        llm = GeminiLLM()

        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful AI assistant that answers questions about YouTube videos 
            using the transcript.
            
            Answer the question: {question}
            Using the following video transcript: {docs}
            
            If there isn't enough information, reply "I don't know."
            
            Your response should be clear and informative.
            """,
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(question=query, docs=docs_page_content)
        return response.strip(), docs

    except Exception as e:
        return f"🚨 Error while generating response: {str(e)}", []
