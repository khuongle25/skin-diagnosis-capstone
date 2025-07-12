import os
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionRAG:
    def __init__(self, num_results=5):
        self.num_results = num_results
    
    def get_context(self, lesion_type):
        """Get context information about a specific skin lesion type"""
        # Giữ nguyên logic hiện tại
        pass
    
    def _compile_context(self, contexts):
        """Compile context information into a single string"""
        if not contexts:
            return "No specific information available about this skin lesion type."
        
        compiled_context = ""
        for i, context in enumerate(contexts):
            compiled_context += f"Source {i+1}: {context['title']} ({context['url']})\n"
            compiled_context += f"{context['content'][:800]}...\n\n"
        
        return compiled_context

class SkinLesionChatbot:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Khởi tạo OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.rag_system = SkinLesionRAG()
    
    def get_response(self, user_message, lesion_type=None, conversation_history=None):
        """
        Get response from OpenAI GPT-3.5 based on user message and lesion type context
        """
        try:
            # Lấy context về loại tổn thương
            context = ""
            if lesion_type:
                context_data = self.rag_system.get_context(lesion_type)
                if context_data:
                    context = self.rag_system._compile_context(context_data)
            
            # Tạo system prompt
            system_prompt = f"""You are a helpful medical AI assistant specializing in dermatology and skin lesions. 
You provide educational information about skin conditions but always remind users to consult healthcare professionals for proper diagnosis and treatment.

Context about {lesion_type if lesion_type else 'skin lesions'}:
{context}

Guidelines:
- Provide accurate, helpful information about skin conditions
- Always recommend consulting a dermatologist for proper diagnosis
- Be empathetic and supportive
- Explain medical terms in simple language
- Do not provide specific medical advice or treatment recommendations
"""

            # Tạo messages cho conversation
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Thêm lịch sử hội thoại nếu có
            if conversation_history:
                for entry in conversation_history[-10:]:  # Chỉ lấy 10 tin nhắn gần nhất
                    messages.append({"role": "user", "content": entry.get("user", "")})
                    messages.append({"role": "assistant", "content": entry.get("assistant", "")})
            
            # Thêm tin nhắn hiện tại
            messages.append({"role": "user", "content": user_message})
            
            # Gọi OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )
            
            assistant_response = response.choices[0].message.content
            
            logger.info(f"Generated response with {len(assistant_response)} characters")
            return assistant_response
            
        except Exception as e:
            logger.error(f"Generate content error: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
        
