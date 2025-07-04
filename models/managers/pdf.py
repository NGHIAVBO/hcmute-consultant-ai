import os
import time
from PyPDF2 import PdfReader
from config import PDF_FILE

def process_directory_pdfs(force_reprocess=False, get_text_chunks_fn=None, get_vector_database_fn=None):
    try:
        if os.path.isabs(PDF_FILE):
            pdf_path = PDF_FILE
        else:
            pdf_path = os.path.join("data-chatbot", PDF_FILE)
        
        if not os.path.exists(pdf_path):
            return f"Không tìm thấy file {pdf_path}.", False
            
        if not force_reprocess and os.path.exists("faiss_index") and os.path.exists("faiss_index/index.faiss"):
            return f"Đã tải {PDF_FILE} từ bộ nhớ cache.", True
        
        start_time = time.time()
        text_with_metadata = []
        
        try:
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                
                if total_pages == 0:
                    return "File PDF không có trang nào.", False
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_with_metadata.append({
                                "text": text,
                                "metadata": {
                                    "source": PDF_FILE,
                                    "page": i + 1,
                                    "total_pages": total_pages
                                }
                            })
                    except Exception:
                        continue
                        
        except Exception as e:
            return f"Lỗi khi đọc file PDF: {str(e)}", False
        
        if not text_with_metadata:
            return "Không thể trích xuất văn bản từ file PDF.", False
            
        chunks = get_text_chunks_fn(text_with_metadata)
        if not chunks:
            return "Không thể tạo chunks từ văn bản.", False
            
        vectorstore = get_vector_database_fn(chunks)
        
        if not os.path.exists("faiss_index"):
            os.makedirs("faiss_index")
        
        process_time = time.time() - start_time
        return f"Đã xử lý {PDF_FILE} thành công! ({process_time:.2f}s)", True
        
    except Exception as e:
        return f"Lỗi khi xử lý PDF: {str(e)}", False
