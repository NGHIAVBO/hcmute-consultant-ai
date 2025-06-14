"""Unified Gemini helper module with personalization
=================================================
This module centralises all Gemini‑related helpers:
  • RAG over vector DB (PDF knowledge)
  • MySQL Q&A
  • Alternative re‑phrasings
  • **Personalisation stage** to adapt the answer to a given user context
  • Markdown post‑processing helpers (fix over‑bold, tighten spacing)

All public helpers return **plain markdown** ready for chat UI.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Optional, List

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import load_qa_chain

from config import (
    GEMINI_MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    TOP_K,
    TOP_P,
    MAX_RETRIES,
    BASE_DELAY,
    VECTOR_SEARCH_K,
    MAX_DOCS,
)

# --------------------------------------------------------------------
# Fallback message
# --------------------------------------------------------------------
_FALLBACK_TEMPLATE = (
    "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại "
    "nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp "
    "chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question?content={question}' "
    "class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. "
    "Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. "
    "Rất mong bạn thông cảm."
)

def _fallback(question: str = "") -> str:
    return _FALLBACK_TEMPLATE.format(question=question)

# --------------------------------------------------------------------
# Text cleanup utilities
# --------------------------------------------------------------------
_OVERBOLD_RE = re.compile(r"^\*\*(.+)\*\*$", re.DOTALL)
_GREETING_NEWLINE_RE = re.compile(r"\*\*Chào bạn,\*\*\n\n")
_DEF_CLEAN_RE = re.compile(r"Dựa trên thông tin trong.+?\\.pdf[:,]?\\s*", flags=re.I)

def fix_overbold_response(text: str) -> str:
    m = _OVERBOLD_RE.match(text.strip())
    if m and "," in m.group(1):
        first, rest = m.group(1).split(",", 1)
        return f"**{first.strip()},**\n{rest.strip()}"
    return text

def tighten_greeting_spacing(text: str) -> str:
    return _GREETING_NEWLINE_RE.sub("**Chào bạn,**\n", text)

def clean_question(q: str) -> str:
    return _DEF_CLEAN_RE.sub("", q or "").strip()

# --------------------------------------------------------------------
# Personalization
# --------------------------------------------------------------------
PERSONALIZATION_TEMPLATE = """
Bạn là trợ lý AI thân thiện và chuyên nghiệp.

**Mục tiêu**: Điều chỉnh lại câu trả lời cho phù hợp với bối cảnh người dùng.
- Đầu vào của bạn là phần **Trả lời gốc** (original_answer) và **Ngữ cảnh người dùng** (user_context).
- Hãy sửa đổi nội dung sao cho phù hợp với ngữ cảnh mới nhưng **không** được bịa thêm thông tin.
- Giữ nguyên phong cách lịch sự, rõ ràng, bắt đầu bằng "Chào bạn," và kết thúc bằng cụm lời cảm ơn thân thiện.
- Nếu nội dung không liên quan tới ngữ cảnh mới, hãy điều chỉnh ví dụ, tên ngành, thuật ngữ chuyên môn cho phù hợp.

**Ngữ cảnh người dùng**: {user_context}

**Trả lời gốc**:
{original_answer}

**Trả lời sau khi điều chỉnh** (dùng Markdown):
"""

def personalize_answer(original_answer: str, user_context: Optional[str] = None) -> str:
    if not user_context:
        return original_answer
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P,
        )
        prompt = PromptTemplate(
            template=PERSONALIZATION_TEMPLATE,
            input_variables=["user_context", "original_answer"],
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        resp = chain.invoke(
            {"input_documents": [], "user_context": user_context, "original_answer": original_answer},
            return_only_outputs=True,
        )
        return resp["output_text"].strip() or original_answer
    except Exception:
        logging.exception("Personalization stage failed")
        return original_answer

# --------------------------------------------------------------------
# Prompt Templates
# --------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """
Bạn là trợ lý AI thân thiện, chuyên phân tích tài liệu PDF. Trả lời câu hỏi dựa CHỈ vào nội dung tài liệu được cung cấp.

**Quy tắc**:
- Chỉ dùng thông tin từ tài liệu dưới đây.
- Không bịa đặt hoặc thêm thông tin ngoài tài liệu.
- Nếu không có thông tin, trả lời: "Chào bạn, cảm ơn bạn đã gửi câu hỏi đến chúng tôi. Tuy nhiên, hiện tại nội dung câu hỏi nằm ngoài phạm vi hỗ trợ của hệ thống. Để được giải đáp chi tiết hơn, bạn có thể <a href='https://hcmute-consultant.vercel.app/create-question' class='text-primary hover:underline'>đặt câu hỏi tại đây</a> để được tư vấn viên trả lời. Chúng tôi sẽ ghi nhận câu hỏi này và cập nhật thêm dữ liệu để có thể trả lời tốt hơn trong tương lai. Rất mong bạn thông cảm."
- Trả lời thân thiện, đầy đủ nhưng ngắn gọn.
- Không đề cập đến độ tin cậy.
**Hướng dẫn về định dạng**:
- Khi câu kết thúc bằng "bao gồm:", "như là:", "gồm:", "như sau:", "điều sau:" hoặc dấu hai chấm (:), hãy trình bày thông tin tiếp theo dưới dạng danh sách có cấu trúc với bullet points (sử dụng dấu * hoặc -).
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

**Câu hỏi**: {question}

**Trả lời** (dùng Markdown, thân thiện và chi tiết):
"""

MYSQL_PROMPT_TEMPLATE = """
Bạn là trợ lý AI hữu ích trả lời câu hỏi dựa trên nội dung cơ sở dữ liệu.

NỘI DUNG CƠ SỞ DỮ LIỆU (Cặp Câu hỏi-Trả lời):
{context}

CÂU HỎI NGƯỜI DÙNG: {user_question}

Dựa CHỈ vào thông tin trong cơ sở dữ liệu trên, cung cấp câu trả lời phù hợp nhất.
Nếu không có thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi, hãy trả lời fallback.
"""

ALTERNATIVE_PROMPT_TEMPLATE = """
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
CÂU TRẢ LỜI GỐC: {base_answer}

CHỈ TRẢ VỀ 5 CÂU TRẢ LỜI THAY THẾ, MỖI CÂU TRÊN 1 ĐOẠN VĂN, KHÔNG ĐÁNH SỐ, KHÔNG THÊM BẤT KỲ GIẢI THÍCH NÀO KHÁC.
"""

# --------------------------------------------------------------------
# Gemini: RAG PDF
# --------------------------------------------------------------------
def get_gemini_rag(vector_db: FAISS, user_question: str, *, filter_pdf: Optional[str] = None,
                   user_context: Optional[str] = None) -> str:
    question = clean_question(user_question)
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    if filter_pdf:
        docs = [d for d in vector_db.docstore._dict.values() if d.metadata.get("source") == filter_pdf]
        if not docs:
            return _fallback(question)
        selected = docs[:MAX_DOCS]
    else:
        selected = vector_db.similarity_search(question, k=VECTOR_SEARCH_K)[:MAX_DOCS]

    for _ in range(MAX_RETRIES):
        try:
            res = chain.invoke({"input_documents": selected, "question": question}, return_only_outputs=True)
            answer = res["output_text"].strip()
            if not answer or "không tìm thấy" in answer.lower():
                answer = _fallback(question)
            return fix_overbold_response(tighten_greeting_spacing(personalize_answer(answer, user_context)))
        except Exception:
            time.sleep(BASE_DELAY)
    return _fallback(question)

# --------------------------------------------------------------------
# Gemini: MySQL Q&A
# --------------------------------------------------------------------
def get_gemini_mysql(user_question: str, *, user_context: Optional[str] = None) -> str:
    from models.managers.mysql import fetch_data_from_mysql
    df = fetch_data_from_mysql()
    if df.empty:
        return _fallback(user_question)

    context = "\n\n".join(
        f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}" for _, row in df.iterrows()
    )

    prompt = MYSQL_PROMPT_TEMPLATE.format(context=context, user_question=user_question)
    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        answer = getattr(resp, "text", "").strip()
        if not answer or "không tìm thấy" in answer.lower():
            answer = _fallback(user_question)
        return fix_overbold_response(tighten_greeting_spacing(personalize_answer(answer, user_context)))
    except Exception:
        logging.exception("MySQL stage failed")
        return _fallback(user_question)

# --------------------------------------------------------------------
# Gemini: Rewriting Alternatives
# --------------------------------------------------------------------
def get_gemini_alternatives(question: str, base_answer: str) -> List[str]:
    prompt = ALTERNATIVE_PROMPT_TEMPLATE.format(question=question, base_answer=base_answer)
    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        raw = getattr(resp, "text", "").strip().split("\n")
        return [l.strip() for l in raw if l.strip()][:5]
    except Exception:
        logging.exception("Alternative generation failed")
        return []
