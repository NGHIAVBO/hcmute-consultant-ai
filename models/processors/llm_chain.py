import google.generativeai as genai
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from config import (
    GEMINI_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS, 
    TOP_K, TOP_P, MAX_RETRIES, BASE_DELAY, MAX_DOCS,
    VECTOR_SEARCH_K,
)
from models.managers.mysql import fetch_data_from_mysql

def get_gemini_rag(vector_database, user_question, filter_pdf=None):
    """
    Combined RAG (Retrieval Augmented Generation) function using Gemini model
    """
    prompt_template = """
    Bạn là trợ lý AI thân thiện và chuyên nghiệp. 

    **Quy tắc chung**:
    - Bắt đầu câu trả lời bằng "Chào bạn," hoặc các từ ngữ thân thiện tương tự
    - Trả lời một cách rõ ràng, dễ hiểu và chuyên nghiệp
    - Nếu thông tin có nhiều điểm cần liệt kê, sử dụng dấu gạch đầu dòng để trình bày
    - Kết thúc câu trả lời bằng "Cảm ơn câu hỏi của bạn" hoặc "Rất vui được hỗ trợ bạn" hoặc các cụm từ thân thiện tương tự
    - Nếu thông tin trong câu trả lời có bảng biểu, sử dụng định dạng markdown table để trình bày
    - Không đề cập đến độ tin cậy
    - Không đề cập đến nguồn tài liệu trong câu trả lời

    **Quy tắc riêng cho phân tích**:
    - Chỉ dùng thông tin từ tài liệu dưới đây
    - Không bịa đặt hoặc thêm thông tin ngoài tài liệu
    - Nếu không có thông tin, trả lời: "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={user_question}' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm."
    
    **Hướng dẫn về định dạng**:
    - Khi câu kết thúc bằng "bao gồm:", "như là:", "gồm:", "như sau:", "điều sau:" hoặc dấu hai chấm (:), hãy trình bày thông tin tiếp theo dưới dạng danh sách có cấu trúc với bullet points (sử dụng dấu * hoặc -).
    - Đảm bảo thụt đầu dòng các bullet points để tạo cấu trúc phân cấp rõ ràng.
    
    **Xử lý danh sách**:
    - Nhận dạng các dòng bắt đầu bằng "•", "○", "▪", "▫", "►", "➢", "➤", "→", "-" hoặc các dấu tương tự như bullet points.
    - Nhận dạng các dòng bắt đầu bằng "□", "☐", "◯", "○", "⬜" như checkbox chưa chọn.
    - Nhận dạng các dòng bắt đầu bằng "■", "☑", "☒", "●", "⬛" như checkbox đã chọn.
    - Áp dụng cấu trúc phân cấp dựa trên khoảng cách thụt đầu dòng:
      * Nếu dòng thụt vào nhiều hơn so với dòng trên, coi đó là sub-bullet của dòng trên.
      * Nếu một dòng có khoảng cách thụt đầu dòng giống dòng trước, coi chúng cùng cấp.
    - Chuyển đổi sang Markdown bằng cách:
      * Dùng dấu "*" hoặc "-" cho các bullet points.
      * Thụt đầu dòng 2 hoặc 4 khoảng trắng cho sub-bullets.
      * Dùng "- [ ]" cho checkbox chưa chọn và "- [x]" cho checkbox đã chọn.
    
    **Hướng dẫn về bảng**:
    - Khi trình bày dữ liệu dạng bảng, phải sử dụng định dạng Markdown table với cột và hàng rõ ràng.
    - Ghi rõ tiêu đề các cột.
    - Đảm bảo căn chỉnh các cột phù hợp.
    - Không sử dụng bullet points cho dữ liệu bảng.
    - Trình bày đầy đủ giá trị của từng ô trong bảng.
    
    **Ví dụ về định dạng bảng đúng**:
    ```
    | Xếp loại | Điểm số | Điểm quy đổi |
    |----------|---------|--------------|
    | Xuất sắc | 9.50-10 | 18           |
    | Giỏi     | 8.50-8.99 | 16         |
    ```
    
    **Ví dụ về định dạng danh sách có cấu trúc đúng**:
    Các thành phần của quy trình bao gồm:
    * Bước 1: Đăng ký học phần
    * Bước 2: Thanh toán học phí
      * Thanh toán trực tiếp
      * Thanh toán online
    * Bước 3: Xác nhận hoàn tất

    **Tài liệu**: {context}

    **Câu hỏi**: {user_question}

    **Trả lời** (dùng Markdown, thân thiện và chi tiết):
    """

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        if filter_pdf:
            docs = [doc for doc_id, doc in vector_database.docstore._dict.items() if doc.metadata.get("source") == filter_pdf]
            if not docs:
                return {"output_text": "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={question}' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm.", "source_documents": [], "structured_tables": []}
            relevant_docs = docs[:MAX_DOCS]
        else:
            vector_docs = vector_database.similarity_search(user_question, k=VECTOR_SEARCH_K)
            relevant_docs = vector_docs[:MAX_DOCS]
        
        for doc in relevant_docs:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata.setdefault('source', 'không xác định')
            doc.metadata.setdefault('page', 'không xác định')
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                result = chain.invoke({"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
                processed_result = post_process_tables(result["output_text"])
                return {
                    "output_text": processed_result["original_response"], 
                    "source_documents": relevant_docs, 
                    "structured_tables": processed_result["structured_tables"]
                }
            except Exception:
                retries += 1
                if retries == MAX_RETRIES:
                    return {
                        "output_text": "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={question}' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm.", 
                        "source_documents": [], 
                        "structured_tables": []
                    }
                time.sleep(BASE_DELAY)
    except Exception:
        return {
            "output_text": "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={question}' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm.", 
            "source_documents": [], 
            "structured_tables": []
        }

def post_process_tables(response):
    import re
    table_pattern = r'\|[^\n]+\|\n\|[-|\s]+\|\n(\|[^\n]+\|\n)+'
    tables = re.findall(table_pattern, response)
    structured_tables = []
    for table in tables:
        lines = table.strip().split('\n')
        headers = [h.strip() for h in lines[0].split('|')[1:-1]]
        data = []
        for row in lines[2:]:
            if row.strip():
                values = [cell.strip() for cell in row.split('|')[1:-1]]
                row_data = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
                data.append(row_data)
        structured_tables.append({'headers': headers, 'data': data})
    return {'original_response': response, 'structured_tables': structured_tables}

def get_gemini_answer(question, answer):
    """
    Generate alternative answers using Gemini model
    """
    try:
        prompt = f"""
            Bạn là trợ lý AI thân thiện và chuyên nghiệp.

            **Quy tắc chung**:
            - Bắt đầu và kết thúc câu trả lời một cách tự nhiên, như đang trò chuyện thực sự
            - Trả lời rõ ràng, dễ hiểu và chuyên nghiệp
            - Sử dụng ngôn ngữ phù hợp với ngữ cảnh và đối tượng
            - Giữ giọng điệu tự nhiên, không quá formal hay quá thân mật
            - Tránh các cụm từ máy móc hoặc công thức

            **Yêu cầu cho các câu trả lời thay thế**:
            - Tạo 5 cách diễn đạt khác nhau cho cùng một nội dung
            - Mỗi cách diễn đạt nên có một góc nhìn riêng về vấn đề
            - Thay đổi cách tiếp cận nhưng vẫn giữ đúng thông tin cốt lõi
            - Điều chỉnh độ chi tiết phù hợp với từng cách diễn đạt
            - Sử dụng các ví dụ thực tế khi cần thiết để làm rõ ý
            
            CÂU HỎI: {question}
            CÂU TRẢ LỜI GỐC: {answer}
            
            CHỈ TRẢ VỀ 5 CÂU TRẢ LỜI THAY THẾ, MỖI CÂU TRÊN 1 ĐOẠN VĂN, KHÔNG ĐÁNH SỐ, KHÔNG THÊM BẤT KỲ GIẢI THÍCH NÀO KHÁC.
        """
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        )
        if hasattr(response, 'text'):
            raw_answers = []
            current_answer = ""
            for line in response.text.strip().split('\n'):
                if line.strip():
                    if not current_answer:
                        current_answer = line.strip()
                    else:
                        current_answer += " " + line.strip()
                else:
                    if current_answer:
                        raw_answers.append(current_answer)
                        current_answer = ""
            if current_answer:
                raw_answers.append(current_answer)
            return raw_answers
        else:
            return []
    except Exception:
        return []   

def get_gemini_mysql(user_question):
    """
    Get answer from MySQL database using Gemini model
    """
    try:
        qa_data = fetch_data_from_mysql()
        
        if qa_data.empty:
            return None
        
        qa_pairs = []
        for _, row in qa_data.iterrows():
            qa_pairs.append(f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}")
        
        context = "\n\n".join(qa_pairs)
        
        prompt = f"""
        Bạn là trợ lý AI thân thiện và chuyên nghiệp.
        
        **Quy tắc chung**:
        - Bắt đầu câu trả lời bằng "Chào bạn," hoặc các từ ngữ thân thiện tương tự
        - Trả lời một cách rõ ràng, dễ hiểu và chuyên nghiệp
        - Nếu thông tin có nhiều điểm cần liệt kê, sử dụng dấu gạch đầu dòng để trình bày
        - Kết thúc câu trả lời bằng "Cảm ơn câu hỏi của bạn" hoặc "Rất vui được hỗ trợ bạn" hoặc các cụm từ thân thiện tương tự
        - Nếu thông tin trong câu trả lời có bảng biểu, sử dụng định dạng markdown table để trình bày
        - Không đề cập đến độ tin cậy
        - Không đề cập đến nguồn tài liệu trong câu trả lời

        **Quy tắc riêng cho truy vấn cơ sở dữ liệu**:
        - Chỉ sử dụng thông tin có trong cơ sở dữ liệu
        - Không thêm thông tin ngoài cơ sở dữ liệu
        - Nếu không có thông tin liên quan, trả lời: "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={user_question}' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm."
        
        NỘI DUNG CƠ SỞ DỮ LIỆU (Cặp Câu hỏi-Trả lời):
        {context}
        
        CÂU HỎI NGƯỜI DÙNG: {user_question}
        
        Dựa CHỈ vào thông tin trong cơ sở dữ liệu trên, cung cấp câu trả lời phù hợp nhất.
        """
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        )
        
        if hasattr(response, 'text'):
            return response.text.strip()
        else:
            return None
            
    except Exception:
        return None
    