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
genai.configure(api_key=GOOGLE_API_KEY)

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
        return response.text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature
        }

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize Gemini LLM
    llm = GeminiLLM()

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs