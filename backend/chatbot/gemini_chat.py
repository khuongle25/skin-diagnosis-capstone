import os
import logging
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinLesionRAG:
    def __init__(self, num_results=5):
        self.num_results = num_results
    
    def get_context(self, lesion_type):
        """Get context information about a specific skin lesion type"""
        logger.info(f"Searching for information about {lesion_type}")
        from .rag_chain import get_full_lesion_name
        lesion_type = get_full_lesion_name(lesion_type)
        # Search for information about the lesion type
        search_query = f"{lesion_type} skin lesion dermatology"
        search_results = google_search(search_query, self.num_results)
        
        contexts = []
        for result in search_results:
            if "url" in result:
                # Scrape content from the URL
                scraped_content = scrape_website(result["url"])
                if scraped_content and len(scraped_content) > 0:
                    logger.info(f"Added content from {result['url']} to context")
                    contexts.append({
                        "title": result.get("title", ""),
                        "url": result["url"],
                        "content": scraped_content[:2000]  # Giới hạn nội dung để tránh quá tải
                    })
        
        # Compile context information
        return self._compile_context(contexts)
    
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
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Cấu hình Gemini API
        genai.configure(api_key=self.api_key)
        
        # Sử dụng tên model chính xác
        try:
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            # Use gemini flash 8b instead of gemini-1.5-pro:
            # self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
        except Exception as e:
            logger.warning(f"Không thể khởi tạo gemini-1.5-pro: {str(e)}. Thử với gemini-pro...")
            self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
            
        self.rag_system = SkinLesionRAG()
    
    def get_response(self, user_message, lesion_type=None, conversation_history=None):
        """
        Get response from Google Gemini based on user message and lesion type context
        """
        try:
            # Initialize conversation history if None
            if conversation_history is None:
                conversation_history = []
                
            full_lesion_name = lesion_type
            try:
                from .rag_chain import get_full_lesion_name
                full_lesion_name = get_full_lesion_name(lesion_type)
            except Exception:
                pass
            
            # Get context information for the lesion type
            context = ""
            system_prompt = f"""You are a helpful assistant that provides information about skin conditions, 
                particularly about {full_lesion_name if full_lesion_name else lesion_type}.

                The patient has been diagnosed with: **{full_lesion_name if full_lesion_name else lesion_type}**.

                Provide clear, accurate, and helpful information based on medical knowledge.
                Always clarify that you're not a doctor and users should seek professional medical advice.

                IMPORTANT FORMATTING RULES:
                1. Always respond in the SAME LANGUAGE that the user's question is written in.

                2. For BOLD TEXT:
                - Use double asterisks with NO SPACE between the asterisks and text
                - Example: **This is bold text** (correct)
                - NOT: ** This is incorrect ** (has spaces)

                3. For BULLET POINTS:
                - Use a SINGLE asterisk (*) followed by a SINGLE space at the start of the line
                - Add a BLANK LINE between each bullet point
                - Example:
                    * First point
                    
                    * Second point
                    
                    * Third point

                Additional context information:
                {context}
            """
            
            try:
                # Phương pháp 1: Sử dụng generative_content trực tiếp
                prompt = system_prompt + "\n\nUser: " + user_message + "\nAssistant:"
                
                # Thiết lập các tham số generation
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
                
                safety_settings = [
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
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                assistant_response = response.text
                
                logger.info(f"Generated response with {len(assistant_response)} characters")
                return assistant_response
                
            except Exception as e:
                logger.error(f"Generate content error: {str(e)}")
                return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"
        
import requests
from chatbot.rag_chain import SkinLesionRAGChain

class HybridSkinLesionChatbot:
    def __init__(self, api_key, user_id, conversation_id):
        self.api_key = api_key
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.ft_url = "https://e3ce-35-201-128-252.ngrok-free.app/chat"

        genai.configure(api_key=self.api_key)
        try:
            # self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception:
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')

    def get_ft_answer(self, question_vi, lesion_type, history=None):
        from .rag_chain import get_full_lesion_name
        full_lesion_name = get_full_lesion_name(lesion_type)
        
        # GEMINI TỔNG HỢP MEMORY + DỊCH
        if history and len(history) > 0:
            # Có lịch sử - Gemini tổng hợp memory + dịch
            history_text = ""
            for msg in history[-5:]:  # 5 lượt gần nhất
                role = "Người dùng" if msg["role"] == "human" else "AI"
                history_text += f"{role}: {msg['content']}\n"
            
            memory_prompt = f"""
            Bạn là chuyên gia y khoa da liễu AI. NHIỆM VỤ LẦN LƯỢT:

            1. PHÂN TÍCH NGỮ CẢNH: Dựa vào lịch sử hội thoại và câu hỏi mới
            2. TỔNG HỢP THÔNG TIN: Kết hợp thông tin chẩn đoán với ngữ cảnh hội thoại
            3. DỊCH SANG TIẾNG ANH: Tạo câu hỏi tiếng Anh hoàn chỉnh cho mô hình chuyên sâu

            **THÔNG TIN CHẨN ĐOÁN:**
            - Loại tổn thương: {full_lesion_name} 

            **LỊCH SỬ HỘI THOẠI:**
            {history_text}

            **CÂU HỎI MỚI (tiếng Việt):**
            "{question_vi}"

            **YÊU CẦU ĐẦU RA:**
            Tạo câu hỏi tiếng Anh hoàn chỉnh, bao gồm:
            - Thông tin chẩn đoán, phải đảm bảo chủ thể cuộc trò chuyện có chủ thể chính là loại tổn thương {full_lesion_name} 
            - Ngữ cảnh từ lịch sử hội thoại (nếu có thông tin liên quan)
            - Câu hỏi được dịch chính xác ngữ nghĩa y khoa da liễu

            CHỈ trả về câu hỏi tiếng Anh cuối cùng, KHÔNG giải thích:
            """
            
            try:
                response = self.gemini_model.generate_content(memory_prompt)
                # print(f"Gemini memory synthesis response: {response.text}")
                logger.info(f"Gemini memory synthesis response: {response.text}")
                question_with_context = response.text.strip()
            except Exception as e:
                logger.error(f"Gemini memory synthesis error: {str(e)}")
                # Fallback: dịch đơn giản
                question_en = self.gemini_translate(question_vi)
                question_with_context = f"[Patient diagnosed with: {full_lesion_name} ({lesion_type})] {question_en}"
        else:
            # Không có lịch sử - chỉ dịch đơn giản
            question_en = self.gemini_translate(question_vi)
            question_with_context = f"[Patient diagnosed with: {full_lesion_name} ({lesion_type})] {question_en}"
        
        # Gửi đến fine-tuned model
        try:
            resp = requests.post(self.ft_url, json={
                "question": question_with_context,
                "lesion_type": lesion_type,
                "lesion_full_name": full_lesion_name
            })
            return resp.json().get("answer", "")
        except Exception as e:
            logger.error(f"Fine-tuned model error: {str(e)}")
            return f"Fine-tuned model unavailable: {str(e)}"

    def get_rag_answer(self, question_vi, lesion_type, enhance_knowledge=False):
        rag_chain = SkinLesionRAGChain(
            api_key=self.api_key,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        # Nếu vectorstore chưa tồn tại, khởi tạo trước (blocking)
        if not rag_chain.vectorstore:
            initialized = rag_chain.initialize_knowledge(lesion_type)
            if not initialized or not rag_chain.vectorstore:
                # Nếu scrape lỗi hoặc vectorstore vẫn chưa có, trả về lỗi
                return "Không thể lấy dữ liệu web do lỗi khởi tạo kiến thức."
        # Đến đây chắc chắn đã scrape xong và có vectorstore
        return rag_chain.get_response(
            message=question_vi,
            lesion_type=lesion_type,
            enhance_knowledge=enhance_knowledge
        )

    def gemini_translate(self, text, direction="vi_to_en"):
        """
        Dịch thuật chuyên nghiệp
        direction: "vi_to_en" hoặc "en_to_vi"
        """
        if direction == "vi_to_en":
            prompt = (
                "Bạn là một chuyên gia dịch thuật y khoa chuyên ngành da liễu. Nhiệm vụ của bạn là:\n"
                "- Dịch chính xác câu hỏi từ tiếng Việt sang tiếng Anh.\n"
                "- Giữ nguyên thuật ngữ y khoa da liễu, không dịch sai hoặc làm mất ý nghĩa chuyên ngành.\n"
                "- Không dịch theo nghĩa đen, mà phải hiểu đúng ngữ cảnh người bệnh.\n"
                f'Đầu vào tiếng Việt:\n"{text}"\n'
                "Đầu ra mong muốn (tiếng Anh, chính xác ngữ nghĩa y khoa da liễu):"
            )
        else:  # en_to_vi
            prompt = (
                "Bạn là một chuyên gia dịch thuật y khoa chuyên ngành da liễu. Nhiệm vụ của bạn là:\n"
                "- Dịch chính xác câu trả lời từ tiếng Anh sang tiếng Việt.\n"
                "- Giữ nguyên thuật ngữ y khoa da liễu, không dịch sai hoặc làm mất ý nghĩa chuyên ngành.\n"
                "- Dịch tự nhiên, dễ hiểu cho bệnh nhân Việt Nam.\n"
                "- CHỈ trả về bản dịch tiếng Việt, KHÔNG giải thích.\n"
                f'Đầu vào tiếng Anh:\n"{text}"\n'
                "Đầu ra mong muốn (tiếng Việt, tự nhiên, dễ hiểu):"
            )
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Translation error ({direction}): {str(e)}")
            return text
        
    def evaluate_and_fix_answer(self, question, ft_answer_en, lesion_type):
            """
            Đánh giá mức độ liên quan giữa câu hỏi và câu trả lời từ mô hình fine-tune.
            Nếu câu trả lời lạc đề hoặc không trả lời trực tiếp, Gemini sẽ tạo lại câu trả lời.
            """
            eval_prompt = f"""
            
            Bạn là một chuyên gia y khoa da liễu AI. Nhiệm vụ của bạn là đánh giá tính liên quan của câu trả lời.
            - Nếu câu trả lời dù không hoàn hảo nhưng vẫn liên quan và có tồn tại dù chỉ một ý trả lời đúng vào trọng tâm câu hỏi, hãy trả lời: "OK".
            - Nếu câu trả lời lạc đề, chung chung hoặc chỉ bảo người dùng kiểm tra phần khác thay vì trả lời, hãy trả lời: "FIX".
            KHÔNG giải thích gì thêm.
            
            Question: "{question}"

            Answer: "{ft_answer_en}"
            """
            try:
                eval_result = self.gemini_model.generate_content(eval_prompt).text.strip()
            except Exception as e:
                logger.error(f"Gemini evaluation error: {str(e)}")
                return ft_answer_en, False  

            if eval_result.strip().upper() == "OK":
                return ft_answer_en, False  

            fix_prompt = f"""
            You are a medical assistant. The patient has been diagnosed with: {lesion_type}.
            Please answer the following question in a helpful, direct, and informative way for the patient, based on medical knowledge. Do not refer the user to other sections or resources. Do not mention you are an AI.

            Question: "{question}"
            """
            try:
                fixed_answer = self.gemini_model.generate_content(fix_prompt).text.strip()
                # print("=== [Gemini Intervene] ===")
                # print(f"Original FT answer: {ft_answer_en}")
                # print(f"Gemini fixed answer: {fixed_answer}")
                # print("==========================")
                logger.info(f"Gemini fixed answer: {fixed_answer}")
                logger.info(f"Original FT answer: {ft_answer_en}")
                logger.info("======================")    
                return fixed_answer, True
            except Exception as e:
                logger.error(f"Gemini fix error: {str(e)}")
                return ft_answer_en, False

    def synthesize_final_answer(self, question_vi, ft_answer_vi, rag_answer, lesion_type):
        """
        Tổng hợp hai nguồn trả lời (mô hình chuyên sâu + RAG) thành một đoạn ngắn gọn, dễ hiểu, ưu tiên thông tin y khoa chính xác.
        Trả về tiếng Việt, không lặp lại nguyên văn, không giải thích thêm.
        """
        from .rag_chain import get_full_lesion_name
        full_lesion_name = get_full_lesion_name(lesion_type)
        prompt = f"""
        Bạn là trợ lý y khoa AI chuyên da liễu. Hãy tổng hợp thông tin từ hai nguồn trả lời dưới đây thành một đoạn ngắn gọn, dễ hiểu, ưu tiên thông tin y khoa chính xác, dành cho bệnh nhân Việt Nam.
        - Không lặp lại nguyên văn từng đoạn. Tuyệt đối không tự ý thêm thông tin ngoài hai nguồn.
        - Nếu có mâu thuẫn, hãy chỉ rõ mâu thuẫn đó giữa hai nguồn trả lời.
        - Chỉ trả về đoạn tổng hợp, không giải thích thêm.
        - Trả lời bằng tiếng Việt.

        **Chẩn đoán:** {full_lesion_name} ({lesion_type})
        **Câu hỏi:** {question_vi}

        **Trả lời từ mô hình chuyên sâu:**
        {ft_answer_vi}

        **Trả lời từ web y khoa (RAG):**
        {rag_answer if rag_answer else 'Không có thông tin bổ sung từ web.'}

        ===
        Đoạn tổng hợp dành cho bệnh nhân:
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini synthesis error: {str(e)}")
            return "Không thể tổng hợp thông tin do lỗi hệ thống. Vui lòng tham khảo các phần thông tin bên dưới."

    def chat(self, question_vi, lesion_type, enhance_knowledge=False, history=None):
            from .rag_chain import get_full_lesion_name
            full_lesion_name = get_full_lesion_name(lesion_type)

            # 1. Fine-tuned model (Gemini tổng hợp memory + dịch) - TRUYỀN question_vi và history
            ft_answer_en = self.get_ft_answer(question_vi, lesion_type, history=history)

            # Đánh giá và can thiệp nếu cần
            ft_answer_en_checked, intervened = self.evaluate_and_fix_answer(
                question_vi, ft_answer_en, full_lesion_name
            )

            # 2. Dịch ft_answer từ Anh → Việt
            ft_answer_vi = ""
            if ft_answer_en_checked and ft_answer_en_checked.strip():
                ft_answer_vi = self.gemini_translate(ft_answer_en_checked, direction="en_to_vi")

            # 3. RAG answer (nếu enhance_knowledge=True)
            rag_answer = ""
            if enhance_knowledge:
                rag_answer = self.get_rag_answer(question_vi, lesion_type, enhance_knowledge=True)

            # 4. SYNTHESIS: Chỉ tổng hợp nếu có enhance_knowledge và rag_answer
            diagnosis_info = f"🩺 **Chẩn đoán**: {full_lesion_name} ({lesion_type})\n\n"
            final_answer = ""
            if enhance_knowledge and rag_answer:
                synthesis = self.synthesize_final_answer(question_vi, ft_answer_vi, rag_answer, lesion_type)
                if synthesis:
                    final_answer += f"**🤖 Thông tin tổng hợp:**\n{synthesis}\n\n"
            # 5. SIMPLE CONCATENATION (giữ nguyên)
            final_answer += diagnosis_info
            final_answer += f"**❓ Câu hỏi**: {question_vi}\n\n"
            if ft_answer_vi:
                final_answer += f"**📚 Thông tin từ mô hình chuyên sâu:**\n{ft_answer_vi}\n\n"
            if enhance_knowledge:
                if rag_answer:
                    final_answer += f"**🌐 Thông tin bổ sung từ web y khoa:**\n{rag_answer}\n\n"
                else:
                    final_answer += f"**🌐 Thông tin từ web:**\n⚠️ Không thể lấy thông tin từ web.\n\n"
            else:
                final_answer += f"**🌐 Thông tin từ web:**\n💡 *Bật 'Tìm kiếm Internet' để có thêm thông tin*\n\n"
            final_answer += "⚠️ **Lưu ý**: Thông tin chỉ mang tính tham khảo. Vui lòng tham khảo bác sĩ chuyên khoa."

            return final_answer, ft_answer_en_checked, rag_answer
