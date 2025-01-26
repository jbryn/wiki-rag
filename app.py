import streamlit as st
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging function with timestamp
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"[{timestamp}] [{level}] {message}")

def generate_summary(text, title, chat):
    """Generates text summary using Groq model"""
    try:
        prompt = f"""
        Read the following text and create a concise summary in English.
        Try to preserve the most important information and main points.
        
        Text to summarize:
        {text}
        """
        
        response = chat.invoke(prompt)
        return {
            'title': title,
            'summary': response.content,
            'original_length': len(text),
            'status': 'ok'
        }
    except Exception as e:
        return {
            'title': title,
            'summary': '',
            'original_length': len(text),
            'status': 'error',
            'error': str(e)
        }

def generate_and_save_summaries(articles, chat):
    """Generates summaries for list of articles and saves them to cache"""
    log_message("Starting summary generation...")
    summaries = []
    
    for article in articles:
        log_message(f"Generating summary for: {article['title']}")
        summary = generate_summary(article['content'], article['title'], chat)
        summaries.append(summary)
        
        # Save progress continuously to avoid losing work
        with open('summaries_status.json', 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    log_message(f"Generated {len(summaries)} summaries")
    return summaries

def load_articles():
    """Loads articles from file"""
    try:
        with open('articles.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_message(f"Error loading articles: {str(e)}", "ERROR")
        return None

def load_summaries_from_cache():
    """Loads summaries from cache file"""
    try:
        with open('summaries_status.json', 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            log_message(f"Loaded {len(cached_data)} summaries from cache")
            
            # Convert to Document format
            summary_documents = []
            for item in cached_data:
                if item['status'] == 'ok':  # Use only successfully generated summaries
                    summary_documents.append(
                        Document(
                            page_content=item['summary'],
                            metadata={
                                "title": item['title'],
                                "original_length": item['original_length']
                            }
                        )
                    )
            return summary_documents
    except FileNotFoundError:
        log_message("Cache file not found", "WARNING")
        return None
    except Exception as e:
        log_message(f"Error loading cache: {str(e)}", "ERROR")
        return None

def initialize_system():
    """Initializes RAG system"""
    try:
        log_message("Starting system initialization...")
        
        # Check if API key is available
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            log_message("Groq API key not found. Check .env file", "ERROR")
            return None, None
            
        # Initialize Groq model
        log_message("Initializing Groq model...")
        os.environ['GROQ_API_KEY'] = api_key
        chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
        
        # Load summaries from cache
        summary_documents = load_summaries_from_cache()
        
        # If no summaries in cache, generate them
        if not summary_documents:
            log_message("No summaries in cache, generating new ones...")
            articles = load_articles()
            if not articles:
                log_message("Failed to load articles", "ERROR")
                return None, None
                
            summaries = generate_and_save_summaries(articles, chat)
            summary_documents = [
                Document(
                    page_content=item['summary'],
                    metadata={
                        "title": item['title'],
                        "original_length": item['original_length']
                    }
                )
                for item in summaries if item['status'] == 'ok'
            ]
        
        # Initialize embeddings and FAISS
        log_message("Initializing embeddings model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name='jinaai/jina-embeddings-v2-base-en',
            model_kwargs={'trust_remote_code': True}
        )
        
        log_message("Creating FAISS index...")
        start_time = datetime.now()
        vector_store = FAISS.from_documents(summary_documents, embedding_model)
        time_taken = (datetime.now() - start_time).total_seconds()
        log_message(f"FAISS index created in {time_taken:.2f} seconds")
        
        retriever = vector_store.as_retriever()
        
        # Prepare RAG
        log_message("Preparing RAG system...")
        rag_prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        
        Answer in English in several concise points.
        """)
        
        document_chain = create_stuff_documents_chain(chat, rag_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        log_message("RAG system ready to use", "SUCCESS")
        
        return retrieval_chain, retriever
        
    except Exception as e:
        log_message(f"Error during initialization: {str(e)}", "ERROR")
        raise e

def get_answer_with_sources(question, retrieval_chain, retriever):
    """Function to get answer and sources"""
    try:
        log_message(f"Processing question: {question}")
        
        start_time = datetime.now()
        response = retrieval_chain.invoke({"input": question})
        retrieved_docs = retriever.get_relevant_documents(question)
        process_time = (datetime.now() - start_time).total_seconds()
        
        log_message(f"Found {len(retrieved_docs)} relevant documents")
        log_message(f"Processing time: {process_time:.2f} seconds")
        
        return {
            'answer': response['answer'],
            'sources': [
                {
                    'title': doc.metadata['title'],
                    'content': doc.page_content[:200] + "..."
                } for doc in retrieved_docs
            ]
        }
    except Exception as e:
        log_message(f"Error processing question: {str(e)}", "ERROR")
        raise e

# Page configuration
st.set_page_config(
    page_title="RAG System - Wikipedia",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
    st.session_state.retriever = None

# User interface
st.title("🤖 RAG System - Wikipedia Knowledge Base")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses RAG (Retrieval-Augmented Generation) system 
    to answer questions based on Wikipedia articles.
    
    ### How it works
    1. Enter your question in the text field
    2. System finds the most relevant articles
    3. You receive an answer with sources
    
    ### Technologies
    - LangChain
    - Groq LLM
    - FAISS
    - HuggingFace Embeddings
    """)

# System initialization on first run
if st.session_state.retrieval_chain is None:
    log_message("Starting first system initialization...")
    with st.spinner("Initializing system... (this may take a few minutes)"):
        try:
            retrieval_chain, retriever = initialize_system()
            st.session_state.retrieval_chain = retrieval_chain
            st.session_state.retriever = retriever
            log_message("System initialized successfully!", "SUCCESS")
        except Exception as e:
            log_message(f"Initialization error: {str(e)}", "ERROR")

# Main interface
question = st.text_input("Ask a question:", placeholder="e.g., What is philosophy?")

if st.button("Ask") and question:
    log_message(f"Received new question: {question}")
    with st.spinner("Looking for answer..."):
        try:
            result = get_answer_with_sources(
                question, 
                st.session_state.retrieval_chain,
                st.session_state.retriever
            )
            
            # Display answer
            st.header("Answer:")
            st.write(result['answer'])
            
            # Display sources
            st.header("Sources:")
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"{i}. {source['title']}"):
                    st.write(source['content'])
            
            log_message("Answer generated successfully", "SUCCESS")
                    
        except Exception as e:
            log_message(f"Error generating answer: {str(e)}", "ERROR")