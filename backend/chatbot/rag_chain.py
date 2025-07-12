import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
from typing import Dict, Any, Optional, List
import json
import uuid
from datetime import datetime
import requests
import gc
import psutil
# Google Generative AI
from pydantic import Field
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
# from sentence_transformers import SentenceTransformer
from .models import get_embeddings
from .scraper import scrape_websites_for_query
from .db_utils import save_conversation, get_conversation_by_id, create_tables, get_conversations_by_user
from .memory_config import (
    get_dynamic_config, 
    get_system_memory_info, 
    get_optimal_chunk_size, 
    get_optimal_batch_size,
    should_use_fallback_model,
    should_force_cleanup,
    get_cleanup_strategy,
    get_cleanup_config,
    MAX_MEMORY_USAGE_PERCENT,
    MAX_TEXT_LENGTH,
    ENABLE_MEMORY_MONITORING,
    ENABLE_BATCH_PROCESSING,
    ENABLE_MEMORY_CLEANUP,
    ENABLE_FORCE_CLEANUP,
    ENABLE_AUTO_CLEANUP,
    LOG_MEMORY_USAGE,
    LOG_CLEANUP_EVENTS
)
from contextlib import contextmanager

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Lấy thông tin sử dụng RAM hiện tại"""
    memory = psutil.virtual_memory()
    return {
        'percent': memory.percent,
        'available': memory.available,
        'total': memory.total,
        'used': memory.used
    }

def check_memory_limit():
    """Kiểm tra xem có vượt quá giới hạn RAM không"""
    if not ENABLE_MEMORY_MONITORING:
        return False
        
    memory_info = get_memory_usage()
    if memory_info['percent'] > MAX_MEMORY_USAGE_PERCENT:
        if LOG_MEMORY_USAGE:
            logger.warning(f"Memory usage high: {memory_info['percent']}%")
        return True
    return False

def cleanup_memory():
    """Dọn dẹp bộ nhớ triệt để"""
    if not ENABLE_MEMORY_CLEANUP:
        return
        
    # Force garbage collection
    gc.collect()
    
    # Thêm một số bước dọn dẹp bổ sung
    try:
        # Clear Python's internal caches
        import sys
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
        
        # Clear any remaining references
        gc.collect()
        
        if LOG_MEMORY_USAGE:
            memory_before = get_memory_usage()
            logger.info(f"Memory cleanup completed. RAM: {memory_before['percent']:.1f}%")
            
    except Exception as e:
        logger.warning(f"Error during memory cleanup: {str(e)}")

def force_cleanup_memory():
    """Dọn dẹp bộ nhớ mạnh mẽ hơn với cấu hình động"""
    if not ENABLE_FORCE_CLEANUP:
        return
        
    try:
        config = get_cleanup_config()
        strategy = get_cleanup_strategy()
        
        if LOG_CLEANUP_EVENTS:
            logger.info(f"Starting {strategy} memory cleanup...")
        
        # Multiple garbage collection passes
        for i in range(config['gc_passes']):
            gc.collect()
            if LOG_CLEANUP_EVENTS and i > 0:
                logger.debug(f"GC pass {i+1}/{config['gc_passes']}")
        
        # Clear cache if needed
        if config['clear_cache']:
            import sys
            if hasattr(sys, 'exc_clear'):
                sys.exc_clear()
            
        
        # Clear TensorFlow session if needed
        if config['force_tensorflow_cleanup']:
            try:
                import tensorflow as tf
                if hasattr(tf, 'keras'):
                    tf.keras.backend.clear_session()
                if hasattr(tf, 'compat'):
                    tf.compat.v1.reset_default_graph()
            except Exception as e:
                logger.debug(f"TensorFlow cleanup failed: {str(e)}")
        
        # Final garbage collection
        gc.collect()
        
        if LOG_CLEANUP_EVENTS:
            memory_after = get_memory_usage()
            logger.info(f"{strategy.capitalize()} cleanup completed. RAM: {memory_after['percent']:.1f}%")
        
    except Exception as e:
        logger.warning(f"Error during force cleanup: {str(e)}")

def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Cắt ngắn văn bản nếu quá dài"""
    if len(text) <= max_length:
        return text
    
    # Cắt từ cuối và thêm dấu hiệu
    truncated = text[:max_length] + "... [truncated]"
    logger.warning(f"Text truncated from {len(text)} to {len(truncated)} characters")
    return truncated

def process_documents_in_batches(documents: List[Document], embeddings_model, batch_size: int = None):
    """Xử lý documents theo batch để tiết kiệm RAM"""
    if not ENABLE_BATCH_PROCESSING:
        return documents
        
    if batch_size is None:
        batch_size = get_optimal_batch_size()
        
    processed_docs = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Kiểm tra RAM trước khi xử lý batch
        if check_memory_limit():
            logger.warning("Memory limit reached, cleaning up...")
            cleanup_memory()
        
        # Xử lý batch
        for doc in batch:
            # Cắt ngắn nội dung nếu cần
            doc.page_content = truncate_text(doc.page_content)
            processed_docs.append(doc)
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
    
    return processed_docs

def get_full_lesion_name(lesion_code):
    """Chuyển đổi mã tổn thương thành tên đầy đủ"""
    mapping = {
        'MEL': 'Melanoma',
        'NV': 'Melanocytic Nevi', 
        'BCC': 'Basal Cell Carcinoma',
        'AK': 'Actinic Keratosis',
        'BKL': 'Benign Keratosis-like Lesions',
        'DF': 'Dermatofibroma',
        'SCC': 'Squamous Cell Carcinoma',
        'UNK': 'Unknown',
        'VASC': 'Vascular Lesions'
    }
    return mapping.get(lesion_code, lesion_code)

# Tạo CustomGeminiChatModel thay cho ChatGoogleGenerativeAI
class CustomGeminiChatModel(BaseChatModel):
    """Custom class để sử dụng Google Generative AI với LangChain"""
    
    model_name: str = Field(default="gemini-1.5-pro", description="Tên của mô hình Gemini")
    temperature: float = Field(default=0.7, description="Thông số temperature cho model")
    top_p: float = Field(default=0.95, description="Thông số top-p cho model")
    top_k: int = Field(default=40, description="Thông số top-k cho model")
    max_output_tokens: int = Field(default=1024, description="Số lượng token tối đa cho output")
    api_key: Optional[str] = Field(default=None, description="Google API key")
    
    client: Any = None
    
    # Định nghĩa _llm_type trực tiếp như một property của class
    @property
    def _llm_type(self) -> str:
        """Trả về loại LLM"""
        return "gemini"
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Khởi tạo model với các thông số cấu hình"""
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Cấu hình Gemini API
        genai.configure(api_key=self.api_key)
        
        # Khởi tạo model
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Gọi model để sinh văn bản"""
        prompt = self._convert_messages_to_prompt(messages)
        
        # Gọi Gemini API
        try:
            response = self.client.generate_content(prompt)
            
            # Tạo đúng định dạng mà LangChain mong đợi
            ai_message = AIMessage(content=response.text)
            generation = {"text": response.text}  # Thêm dòng này
            return {"generations": [[generation]], "llm_output": None}  # Thay đổi tại đây
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            ai_message = AIMessage(content=f"Error: {str(e)}")
            generation = {"text": f"Error: {str(e)}"}
            return {"generations": [[generation]], "llm_output": None}
    
    def _convert_messages_to_prompt(self, messages):
        """Chuyển đổi LangChain messages thành prompt Gemini"""
        # Đơn giản hóa: chỉ lấy nội dung message cuối (câu hỏi người dùng)
        if messages and isinstance(messages[-1], HumanMessage):
            return messages[-1].content
        
        # Xử lý trường hợp nhiều messages
        prompt_parts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_parts.append({"role": "user", "parts": [message.content]})
            elif isinstance(message, AIMessage):
                prompt_parts.append({"role": "model", "parts": [message.content]})
        
        return prompt_parts
    
    def invoke(self, input, **kwargs):
        """Invoke the model with a prompt or messages"""
        try:
            if isinstance(input, str):
                prompt = input
            elif isinstance(input, list):
                prompt = self._convert_messages_to_prompt(input)
            else:
                raise ValueError(f"Unsupported input type: {type(input)}")
                
            # Gọi API trực tiếp
            response = self.client.generate_content(prompt)
            return AIMessage(content=response.text)
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            return AIMessage(content=f"Error: {str(e)}")
    
class SkinLesionRAGChain:
    def __init__(self, api_key=None, user_id=None, conversation_id=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.user_id = user_id
        self.conversation_id = conversation_id or str(uuid.uuid4())
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Sử dụng custom Gemini model thay vì ChatGoogleGenerativeAI
        self.llm = CustomGeminiChatModel(
            model_name="gemini-1.5-pro",
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
            api_key=self.api_key
        )
        
        # Lazy loading cho embeddings - chỉ tải khi cần
        self._embeddings = None
        self._embeddings_loaded = False
        
        # Khởi tạo vector store
        base_dir = "chroma_db"
        self.persist_directory = os.path.join(base_dir, str(self.conversation_id))
        
        # Khởi tạo memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Khôi phục lịch sử hội thoại từ database vào memory
        if self.conversation_id:
            try:
                conversation = get_conversation_by_id(self.conversation_id)
                if conversation:
                    # Lấy lịch sử từ database
                    history = json.loads(conversation.get('history', '[]'))
                    
                    # Khôi phục memory từ lịch sử
                    for msg in history:
                        if msg['role'] == 'human':
                            self.memory.chat_memory.add_user_message(msg['content'])
                        elif msg['role'] == 'ai':
                            self.memory.chat_memory.add_ai_message(msg['content'])
                    
                    logger.info(f"Restored conversation memory with {len(history)} messages")
            except Exception as e:
                logger.error(f"Error restoring conversation memory: {str(e)}")
        
        # Khởi tạo chain nếu đã có conversation_id và vector store
        if conversation_id and os.path.exists(self.persist_directory):
            self._load_embeddings()
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._embeddings
            )
            self._create_chain()
            
            # Khôi phục lịch sử hội thoại từ database
            self._restore_conversation_history()
        else:
            self.vectorstore = None
            self.chain = None
    
    def _load_embeddings(self):
        """Lazy loading cho embeddings - chỉ tải khi cần"""
        if self._embeddings_loaded and self._embeddings is not None:
            return self._embeddings
            
        try:
            logger.info("Loading embeddings model...")
            
            # Lấy cấu hình động
            config = get_dynamic_config()
            logger.info(f"System memory: {config['memory_info']}")
            
            # Kiểm tra RAM trước khi tải model
            if check_memory_limit():
                logger.warning("Memory usage high before loading embeddings, cleaning up...")
                cleanup_memory()
            
            # Kiểm tra xem có nên sử dụng fallback model không
            if should_use_fallback_model():
                logger.warning("System memory low, using fallback embedding model")
                from langchain_community.embeddings import TensorflowHubEmbeddings
                self._embeddings = TensorflowHubEmbeddings(
                    model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
                )
            else:
                # Sử dụng model local
                self._embeddings = get_embeddings()
            
            self._embeddings_loaded = True
            logger.info("Embeddings loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            # Fallback: Tải trực tiếp (chỉ trong trường hợp lỗi)
            from langchain_community.embeddings import TensorflowHubEmbeddings
            logger.info("Fallback: Loading embeddings directly from TF Hub")
            self._embeddings = TensorflowHubEmbeddings(
                model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )
            self._embeddings_loaded = True
    
    @property
    def embeddings(self):
        """Property để truy cập embeddings với lazy loading"""
        if not self._embeddings_loaded:
            self._load_embeddings()
        return self._embeddings
    
    def _create_chain(self):
        """Tạo chain RAG với context đã có"""
        if not self.vectorstore:
            raise ValueError("Vector store has not been initialized")
            
        # Template cho system prompt
        system_template = """Bạn là trợ lý y tế hữu ích cung cấp thông tin về các bệnh da liễu, 
        đặc biệt là về: {lesion_type}.
        Cung cấp thông tin rõ ràng, chính xác và hữu ích dựa trên kiến thức y học.
        Luôn làm rõ rằng bạn không phải là bác sĩ và người dùng nên tìm kiếm lời khuyên y tế chuyên nghiệp.
        
        QUY TẮC ĐỊNH DẠNG QUAN TRỌNG:
        1. Luôn trả lời bằng CÙNG NGÔN NGỮ với câu hỏi của người dùng.
        2. Với CHỮ ĐẬM: Sử dụng cú pháp **text** (không có khoảng trắng giữa dấu ** và văn bản)
        3. Với DANH SÁCH: Sử dụng dấu * ở đầu dòng, theo sau là một khoảng trắng
        
        Sử dụng thông tin sau để trả lời (nếu có liên quan):
        {context}
        
        Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. KHÔNG bịa ra thông tin.
        """
        
        # Tạo prompt template
        PROMPT = PromptTemplate(
            template=system_template, 
            input_variables=["context", "lesion_type"]
        )
    
    def _restore_conversation_history(self):
        """Khôi phục lịch sử hội thoại từ database"""
        try:
            # Lấy lịch sử hội thoại từ database
            conversation = get_conversation_by_id(self.conversation_id)
            if conversation and conversation.get('history'):
                history = json.loads(conversation.get('history'))
                for entry in history:
                    if entry['role'] == 'human':
                        self.memory.chat_memory.add_user_message(entry['content'])
                    elif entry['role'] == 'ai':
                        self.memory.chat_memory.add_ai_message(entry['content'])
        except Exception as e:
            logger.error(f"Error restoring conversation history: {str(e)}")
    
    def initialize_knowledge(self, lesion_type: str, client_id: str = None, num_search_results: int = 3):
        try:
            # Chuyển đổi mã thành tên đầy đủ
            full_name = get_full_lesion_name(lesion_type)
            num_search_results = min(num_search_results, 3)
            
            # Kiểm tra nếu vectorstore đã được khởi tạo cho conversation này
            if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
                logger.info(f"Knowledge already initialized for conversation {self.conversation_id}")
                self._load_embeddings()
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self._embeddings
                )
                self._create_chain()
                return True
            
            # Nếu chưa có, tiếp tục khởi tạo mới
            logger.info(f"Initializing new knowledge for lesion type: {lesion_type} ({full_name})")
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Tạo câu truy vấn tìm kiếm chi tiết hơn
            search_queries = [
                f"{full_name} skin lesion dermatology symptoms treatment",
                f"{lesion_type} {full_name} diagnosis medical information",
                f"{full_name} skin cancer dermatology clinical features"
            ]
            
            all_scraped_texts = []
            
            # Scrape với nhiều query khác nhau
            for query in search_queries:
                logger.info(f"Searching with query: {query}")
                scraped_texts = scrape_websites_for_query(
                    query, 
                    num_results=1,  # Ít kết quả hơn cho mỗi query
                    source="initialization"
                )
                all_scraped_texts.extend(scraped_texts)
            
            if not all_scraped_texts:
                logger.warning(f"No information found for lesion type: {lesion_type}")
                # Sử dụng thông tin mặc định chi tiết hơn
                all_scraped_texts = [
                    f"Basic information about {full_name} ({lesion_type}) skin lesions. "
                    f"{full_name} is a type of skin condition that requires medical evaluation. "
                    f"Please consult a dermatologist for proper diagnosis and treatment."
                ]
            
            # Chuyển đổi văn bản scraped thành documents với giới hạn kích thước
            documents = []
            for text in all_scraped_texts:
                if text.strip():
                    # Cắt ngắn văn bản nếu quá dài
                    truncated_text = truncate_text(text, MAX_TEXT_LENGTH)
                    documents.append(Document(page_content=truncated_text))
            
            # Chia nhỏ văn bản với kích thước nhỏ hơn
            chunk_size = get_optimal_chunk_size()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,  # Sử dụng chunk size động
                chunk_overlap=50,  # Giảm overlap
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} document chunks for {full_name} ({lesion_type})")
            
            # Xử lý documents theo batch để tiết kiệm RAM
            processed_docs = process_documents_in_batches(split_docs, None)
            
            # Tải embeddings nếu chưa tải
            if not self._embeddings_loaded:
                self._load_embeddings()
            
            # Tạo vector store mới với batch processing
            logger.info("Creating vector store with batch processing...")
            self.vectorstore = Chroma.from_documents(
                documents=processed_docs,
                embedding=self._embeddings,
                persist_directory=self.persist_directory
            )
            
            # Dọn dẹp bộ nhớ sau khi tạo vector store
            cleanup_memory()
            
            # Tạo chain
            self._create_chain()
            
            # CHỈ lưu conversation rỗng nếu conversation chưa tồn tại
            conversation = get_conversation_by_id(self.conversation_id)
            if not conversation:
                save_conversation(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    lesion_type=lesion_type,
                    history=json.dumps([]),
                    created_at=datetime.now().isoformat()
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing knowledge: {str(e)}")
            return False
            
    def extract_key_points(self, query: str, lesion_type: str, history: Optional[list] = None) -> Dict[str, str]:
        """Trích xuất câu hỏi tìm kiếm song ngữ từ hội thoại"""
        try:
            chat_context = ""
            if history and len(history) > 0:
                for msg in history[-3:]:
                    if isinstance(msg, dict):
                        role = "Người dùng" if msg.get("role") == "human" else "AI"
                        content = msg.get("content", "")
                    else:
                        role = "Người dùng" if getattr(msg, "role", None) == "human" else "AI"
                        content = getattr(msg, "content", "")
                    chat_context += f"{role}: {content}\n"

            # Prompt Gemini để tạo câu hỏi tìm kiếm tiếng Việt
            extraction_prompt = f"""
                Bạn là trợ lý y khoa AI. Nhiệm vụ của bạn là:
                - Dựa trên chủ đề chính: {lesion_type}
                - Dựa trên lịch sử hội thoại sau:
                {chat_context}
                - Và câu hỏi mới nhất: "{query}"

                Hãy tạo ra một câu hỏi tìm kiếm y khoa hoàn chỉnh, rõ ràng, có thể dùng để tìm tài liệu trên Internet, đảm bảo bao quát đầy đủ ý của người dùng và chủ đề chính. 
                Chỉ trả về duy nhất một câu hỏi, không giải thích thêm.
            """
            vi_response = self.llm.invoke(extraction_prompt)
            search_vi = vi_response.content.strip()

            # Dịch sang tiếng Anh (có thể dùng Gemini hoặc Google Translate)
            en_prompt = f"Dịch câu hỏi sau sang tiếng Anh, giữ nguyên ý nghĩa y khoa, không giải thích: \"{search_vi}\""
            en_response = self.llm.invoke(en_prompt)
            search_en = en_response.content.strip()

            logger.info(f"Extracted VI: {search_vi} | EN: {search_en}")
            return {"vi": search_vi, "en": search_en}
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return {"vi": f"{query} {lesion_type} skin condition", "en": f"{query} {lesion_type} skin condition"}
        
    def enhance_knowledge(self, query: str, lesion_type: str, history: Optional[list] = None, num_search_results: int = 3):
        """Bổ sung kiến thức bằng cách tìm kiếm thêm thông tin từ internet"""
        try:
            # Trích xuất ý chính từ câu hỏi (song ngữ)
            search_queries = self.extract_key_points(query, lesion_type, history)
            logger.info(f"Enhanced search queries: {search_queries}")

            num_search_results = min(num_search_results, 3)
            requests.post("http://localhost:8001/reset-events")
            # requests.post("https://sse-server-app-28227929064.us-central1.run.app/reset-events")

            # Scrape tiếng Việt
            new_texts_vi = scrape_websites_for_query(
                search_queries["vi"],
                num_results=num_search_results,
                source="enhancement_vi"
            )
            # Scrape tiếng Anh
            new_texts_en = scrape_websites_for_query(
                search_queries["en"],
                num_results=num_search_results,
                source="enhancement_en"
            )

            # Gộp kết quả song ngữ
            new_texts = new_texts_vi + new_texts_en
            if not new_texts:
                logger.warning(f"No additional information found for query: {query}")
                return False
            
            # Chuyển đổi văn bản mới thành documents với giới hạn kích thước
            new_documents = []
            for text in new_texts:
                if text.strip():
                    # Cắt ngắn văn bản nếu quá dài
                    truncated_text = truncate_text(text, MAX_TEXT_LENGTH)
                    new_documents.append(Document(page_content=truncated_text))
            
            # Chia nhỏ văn bản với kích thước nhỏ hơn
            chunk_size = get_optimal_chunk_size()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=50
            )
            
            new_split_docs = text_splitter.split_documents(new_documents)
            logger.info(f"Created {len(new_split_docs)} additional document chunks")
            
            # Xử lý documents theo batch để tiết kiệm RAM
            processed_new_docs = process_documents_in_batches(new_split_docs, None)
            
            # Kiểm tra RAM trước khi thêm vào vector store
            if check_memory_limit():
                logger.warning("Memory usage high before adding documents, cleaning up...")
                cleanup_memory()
            
            # Thêm documents mới vào vector store hiện tại
            self.vectorstore.add_documents(processed_new_docs)
            
            # Dọn dẹp bộ nhớ sau khi thêm documents
            cleanup_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing knowledge: {str(e)}")
            return False    
        
    def get_response(self, message: str, lesion_type: str, enhance_knowledge: bool = False):
        try:
            if not self.vectorstore:
                logger.error("Vector store has not been initialized")
                return "Xin lỗi, hệ thống chưa khởi tạo kiến thức. Vui lòng thử lại sau."
            
            # Chuyển đổi mã thành tên đầy đủ
            full_lesion_name = get_full_lesion_name(lesion_type)
            
            # Lấy history từ memory
            chat_history = self.memory.load_memory_variables({})["chat_history"]

            # Bổ sung kiến thức nếu được yêu cầu
            if enhance_knowledge:
                logger.info(f"Enhancing knowledge with query: {message}")
                knowledge_updated = self.enhance_knowledge(message, lesion_type, history=chat_history)
                if not knowledge_updated:
                    logger.info("No new knowledge added, using existing knowledge base") 
                        
            # Lưu tin nhắn của người dùng vào bộ nhớ
            self.memory.chat_memory.add_user_message(message)
            
            # Lấy lịch sử trò chuyện từ memory
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # Kiểm tra RAM trước khi truy xuất documents
            if check_memory_limit():
                logger.warning("Memory usage high before retrieval, cleaning up...")
                cleanup_memory()
            
            # Truy xuất documents liên quan với context rõ ràng hơn
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})  # Giảm số lượng documents

            # Tạo query tìm kiếm có context lesion type
            enhanced_message = f"{message} (related to {full_lesion_name} {lesion_type})"
            
            # Truy xuất bằng tiếng Việt với context
            docs_vi = retriever.invoke(enhanced_message)

            # Dịch sang tiếng Anh với context lesion
            en_prompt = f"""Translate this medical question to English, keeping the medical context about {full_lesion_name} ({lesion_type}):
            "{message}"
            
            Only return the English translation, no explanation."""
            
            en_response = self.llm.invoke(en_prompt)
            message_en = en_response.content.strip()
            enhanced_message_en = f"{message_en} (about {full_lesion_name} skin lesion)"

            # Truy xuất bằng tiếng Anh với context
            docs_en = retriever.invoke(enhanced_message_en)

            # Gộp và loại duplicate (theo nội dung) với giới hạn kích thước
            all_docs = {}
            for doc in docs_vi + docs_en:
                # Cắt ngắn nội dung nếu cần
                doc.page_content = truncate_text(doc.page_content, MAX_TEXT_LENGTH // 2)
                all_docs[doc.page_content] = doc
            
            docs = list(all_docs.values())[:3]  # Giảm số lượng documents

            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Tạo prompt với context lesion rõ ràng
            system_template = """Bạn là trợ lý y tế chuyên khoa da liễu đang tư vấn về {full_lesion_name} ({lesion_code}).

            THÔNG TIN CHẨN ĐOÁN:
            - Loại tổn thương: {full_lesion_name}
            - Mã chẩn đoán: {lesion_code}
            
            NHIỆM VỤ:
            - Cung cấp thông tin chính xác về {full_lesion_name}
            - Tư vấn dựa trên kiến thức y học chuyên sâu
            - Luôn nhắc nhở tìm kiếm lời khuyên y tế chuyên nghiệp
            
            QUY TẮC ĐỊNH DẠNG:
            1. Trả lời bằng CÙNG NGÔN NGỮ với câu hỏi
            2. Dùng **text** cho chữ đậm (không có khoảng trắng)
            3. Dùng * ở đầu dòng cho danh sách
            
            THÔNG TIN THAM KHẢO:
            {context}
            
            Nếu không chắc chắn, hãy thành thật nói "tôi không biết".
            
            Câu hỏi về {full_lesion_name}: {question}
            """
            
            # Format prompt với thông tin lesion chi tiết
            formatted_prompt = system_template.format(
                context=context,
                full_lesion_name=full_lesion_name,
                lesion_code=lesion_type,
                question=message
            )
            
            # Gọi LLM
            try:
                llm_response = self.llm.invoke(formatted_prompt)
                answer = llm_response.content
            except Exception as e:
                logger.error(f"Error invoking LLM: {str(e)}")
                answer = f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu về {full_lesion_name}: {str(e)}"
            
            # Lưu câu trả lời vào bộ nhớ
            self.memory.chat_memory.add_ai_message(answer)
            
            # Dọn dẹp bộ nhớ sau khi xử lý xong
            cleanup_memory()
            
            # Clear temporary variables
            del docs_vi, docs_en, all_docs, docs, context, formatted_prompt
            if 'llm_response' in locals():
                del llm_response
            
            # Force cleanup nếu RAM vẫn cao
            memory_after = get_memory_usage()
            if memory_after['percent'] > MAX_MEMORY_USAGE_PERCENT:
                logger.warning(f"Memory still high after response: {memory_after['percent']}%, forcing cleanup...")
                force_cleanup_memory()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu về {get_full_lesion_name(lesion_type)}: {str(e)}"
   
    @staticmethod
    def get_user_conversations(user_id: str):
        """Lấy danh sách các cuộc hội thoại của người dùng"""
        return get_conversations_by_user(user_id)
    
    @staticmethod
    def create_database_tables():
        """Tạo các bảng cần thiết trong database"""
        print("Tạo các bảng cần thiết trong database")
        create_tables()

    def cleanup(self):
        """Dọn dẹp tài nguyên của RAG chain"""
        try:
            logger.info("Cleaning up RAG chain resources...")
            
            # Clear memory
            if hasattr(self, 'memory') and self.memory:
                self.memory.clear()
            
            # Clear vectorstore references
            if hasattr(self, 'vectorstore') and self.vectorstore:
                # Clear any cached embeddings in vectorstore
                if hasattr(self.vectorstore, '_embedding_function'):
                    del self.vectorstore._embedding_function
                self.vectorstore = None
            
            # Clear embeddings model
            if hasattr(self, '_embeddings') and self._embeddings:
                del self._embeddings
                self._embeddings = None
                self._embeddings_loaded = False
            
            # Clear LLM client
            if hasattr(self, 'llm') and hasattr(self.llm, 'client'):
                del self.llm.client
            
            # Force cleanup
            force_cleanup_memory()
            
            logger.info("RAG chain cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during RAG chain cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor để tự động dọn dẹp"""
        try:
            self.cleanup()
        except:
            pass

@contextmanager
def rag_chain_context(api_key=None, user_id=None, conversation_id=None):
    """Context manager để tự động dọn dẹp RAG chain"""
    rag_chain = None
    try:
        rag_chain = SkinLesionRAGChain(
            api_key=api_key,
            user_id=user_id,
            conversation_id=conversation_id
        )
        yield rag_chain
    finally:
        if rag_chain:
            rag_chain.cleanup()