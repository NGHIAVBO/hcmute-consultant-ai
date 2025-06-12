from models.processors.llm_chain import get_gemini_rag
from models.processors.small_talk import is_small_talk
from models.storages.vector_database import load_vector_database
from models.managers.cache import get_cache, set_cache
from models.processors.llm_chain import get_gemini_mysql
from config import PDF_FILE

vector_database = None

def load_vector_db_once():
    global vector_database
    if vector_database is None:
        vector_database = load_vector_database()[0]
    return vector_database

def process_query(prompt):
    cached_result, cache_hit, time_saved = get_cache(prompt)
    if cache_hit:
        return f"{cached_result}\n\n*(Kết quả từ cache, tiết kiệm {time_saved:.2f}s)*"
    
    small_talk_response = is_small_talk(prompt)
    if small_talk_response:
        return small_talk_response
    
    try:
        mysql_result = get_gemini_mysql(prompt)
        if mysql_result and "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu" not in mysql_result:
            set_cache(prompt, mysql_result, 0)
            return mysql_result
        
        vector_database = load_vector_db_once()
        if not vector_database:
            return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."
            
        context_prompt = f"Dựa trên thông tin trong {PDF_FILE}, {prompt}"

        response = get_gemini_rag(vector_database, context_prompt, filter_pdf=PDF_FILE)
        if not response:
            return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."
        
        answer = response["output_text"]
        if not answer:
            return "Xin lỗi, không nhận được câu trả lời. Vui lòng thử lại sau."
            
        if any(phrase in answer.lower() for phrase in ["không tìm thấy thông tin", "không có thông tin"]):
            result = "Không tìm thấy thông tin bạn đưa ra. Vui lòng đặt câu hỏi khác."
        else:
            result = answer
        
        set_cache(prompt, result, 0)
        return result
        
    except Exception as e:
        return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau."
