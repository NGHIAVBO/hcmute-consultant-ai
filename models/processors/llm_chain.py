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


# ==============================================================
# 1. Hàm RAG chính dùng Gemini
# ==============================================================

def get_gemini_rag(vector_database, user_question, filter_pdf=None):
    """
    Combined RAG (Retrieval-Augmented Generation) với Gemini.
    Bảo đảm: có chào-đầu, kết-thúc lịch sự, định dạng Markdown chuẩn,
    giữ nguyên hướng dẫn xử lý danh sách/bảng.
    """

    prompt_template = """
    Bạn là trợ lý AI thân thiện, chuyên phân tích tài liệu PDF. Trả lời câu hỏi dựa CHỈ vào nội dung tài liệu được cung cấp.

    **Quy tắc**:
    - Chỉ dùng thông tin từ tài liệu dưới đây.
    - Không bịa đặt hoặc thêm thông tin ngoài tài liệu.
    - Nếu không có thông tin, trả lời: "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={question}' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm."
    - Trả lời thân thiện, đầy đủ nhưng ngắn gọn.
    - Bắt đầu câu trả lời bằng "Chào bạn," hoặc từ ngữ thân thiện tương tự.
    - Kết thúc câu trả lời bằng: "Cảm ơn câu hỏi của bạn, nếu còn câu hỏi nào vui lòng hỏi để mình giúp bạn trả lời."
    - Không đề cập đến độ tin cậy.

    **Hướng dẫn về định dạng**:
    - Khi câu kết thúc bằng "bao gồm:", "như là:", "gồm:", "như sau:", "điều sau:" hoặc dấu hai chấm (:),
      hãy trình bày nội dung tiếp theo dưới dạng danh sách có cấu trúc (bullet) với "*" hoặc "-".
    - Đảm bảo thụt đầu dòng các bullet points để tạo cấu trúc phân cấp rõ ràng.

    **Xử lý danh sách từ PDF**:
    - Nhận dạng các dòng bắt đầu bằng "•", "○", "▪", "▫", "►", "➢", "➤", "→", "-" hoặc các dấu tương tự như bullet points.
    - Nhận dạng các dòng bắt đầu bằng "□", "☐", "◯", "○", "⬜" như checkbox chưa chọn.
    - Nhận dạng các dòng bắt đầu bằng "■", "☑", "☒", "●", "⬛" như checkbox đã chọn.
    - Áp dụng cấu trúc phân cấp dựa trên khoảng cách thụt đầu dòng:
      * Nếu dòng thụt vào nhiều hơn so với dòng trên, coi đó là sub-bullet của dòng trên.
      * Nếu một dòng có khoảng cách thụt đầu dòng giống dòng trước, coi chúng cùng cấp.
    - Chuyển đổi từ định dạng PDF sang Markdown bằng cách:
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
    | Xếp loại | Điểm số   | Điểm quy đổi |
    |----------|-----------|--------------|
    | Xuất sắc | 9.50-10   | 18           |
    | Giỏi     | 8.50-8.99 | 16           |
    ```

    **Ví dụ về định dạng danh sách có cấu trúc đúng**:
    Các thành phần của quy trình bao gồm:
    * Bước 1: Đăng ký học phần
    * Bước 2: Thanh toán học phí
      * Thanh toán trực tiếp
      * Thanh toán online
    * Bước 3: Xác nhận hoàn tất

    **Tài liệu**: {context}

    **Câu hỏi**: {question}

    **Trả lời** (dùng Markdown, thân thiện, chi tiết):
    """

    # ----------------- Khởi tạo LLM -----------------
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

    # ----------------- Truy xuất tài liệu -----------------
    if filter_pdf:
        docs = [
            doc for _, doc in vector_database.docstore._dict.items()
            if doc.metadata.get("source") == filter_pdf
        ]
    else:
        docs = vector_database.similarity_search(user_question, k=VECTOR_SEARCH_K)

    if not docs:
        return {
            "output_text": (
                "Chào bạn, hiện tại hệ thống không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn. "
                f"Bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={user_question}' "
                "class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên hỗ trợ."
            ),
            "source_documents": [],
            "structured_tables": []
        }

    relevant_docs = docs[:MAX_DOCS]
    for doc in relevant_docs:
        doc.metadata = doc.metadata or {}
        doc.metadata.setdefault('source', 'không xác định')
        doc.metadata.setdefault('page', 'không xác định')

    # ----------------- Gọi LLM & hậu xử lý -----------------
    retries = 0
    while retries < MAX_RETRIES:
        try:
            result = chain.invoke(
                {"input_documents": relevant_docs, "question": user_question},
                return_only_outputs=True
            )
            processed = post_process_tables(result["output_text"])

            # Thêm chào-đầu / kết-thúc nếu thiếu
            final_resp = processed["original_response"].strip()
            if not final_resp.lower().startswith("chào"):
                final_resp = "Chào bạn, " + final_resp
            if "vui lòng hỏi để mình giúp bạn trả lời" not in final_resp.lower():
                final_resp += (
                    "\n\nCảm ơn câu hỏi của bạn, nếu còn câu hỏi nào vui lòng hỏi để mình giúp bạn trả lời."
                )

            return {
                "output_text": final_resp,
                "source_documents": relevant_docs,
                "structured_tables": processed["structured_tables"]
            }

        except Exception:
            retries += 1
            if retries == MAX_RETRIES:
                return {
                    "output_text": "Chào bạn, hệ thống gặp lỗi khi xử lý câu hỏi. Vui lòng thử lại sau.",
                    "source_documents": [],
                    "structured_tables": []
                }
            time.sleep(BASE_DELAY)


# ==============================================================
# 2. Hậu xử lý bảng Markdown thành cấu trúc JSON
# ==============================================================

def post_process_tables(response: str):
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
                cells = [c.strip() for c in row.split('|')[1:-1]]
                data.append({headers[i]: cells[i] for i in range(min(len(headers), len(cells)))})

        structured_tables.append({"headers": headers, "data": data})

    return {"original_response": response, "structured_tables": structured_tables}


# ==============================================================
# 3. Hàm tạo 5 câu trả lời thay thế
# ==============================================================

def get_gemini_answer(question: str, answer: str):
    """
    Tạo 5 câu trả lời thay thế hoàn toàn khác biệt (độ dài, giọng điệu…).
    """
    prompt = f"""
        Dựa vào câu hỏi và câu trả lời gốc dưới đây, hãy tạo chính xác 5 câu trả lời thay thế KHÁC BIỆT HOÀN TOÀN về cách trình bày.
        MỖI câu trả lời PHẢI có:
        - Độ dài khác nhau (ngắn, trung bình, dài)
        - Cách tiếp cận khác nhau (trực tiếp, chi tiết, ví dụ thực tế, hướng dẫn)
        - Giọng điệu khác nhau (trang trọng, thân thiện, chuyên nghiệp, đơn giản)

        CÂU HỎI: {question}
        CÂU TRẢ LỜI GỐC: {answer}

        CHỈ TRẢ VỀ 5 CÂU TRẢ LỜI THAY THẾ, MỖI CÂU TRÊN 1 ĐOẠN VĂN, KHÔNG ĐÁNH SỐ, KHÔNG GIẢI THÍCH THÊM.
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

    if not hasattr(response, "text"):
        return []

    raw_answers, current = [], ""
    for line in response.text.strip().split('\n'):
        if line.strip():
            current += (" " if current else "") + line.strip()
        else:
            if current:
                raw_answers.append(current.strip())
                current = ""
    if current:
        raw_answers.append(current.strip())
    return raw_answers


# ==============================================================
# 4. Truy vấn Q&A từ MySQL bằng Gemini
# ==============================================================

def get_gemini_mysql(user_question: str):
    """
    Trả lời từ cơ sở dữ liệu Q&A MySQL bằng Gemini.
    """
    qa_data = fetch_data_from_mysql()
    if qa_data.empty:
        return None

    qa_pairs = [
        f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}"
        for _, row in qa_data.iterrows()
    ]
    context = "\n\n".join(qa_pairs)

    prompt = f"""
    Bạn là trợ lý AI hữu ích trả lời câu hỏi dựa trên nội dung cơ sở dữ liệu.

    NỘI DUNG CƠ SỞ DỮ LIỆU:
    {context}

    CÂU HỎI NGƯỜI DÙNG: {user_question}

    Dựa CHỈ vào thông tin trên, hãy cung cấp câu trả lời phù hợp nhất.
    Nếu không có thông tin liên quan, trả lời: "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
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
    return response.text.strip() if hasattr(response, "text") else None
