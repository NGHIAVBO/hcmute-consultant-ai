import pandas as pd
import mysql.connector
from mysql.connector import Error, pooling
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from functools import lru_cache
import contextlib
import google.generativeai as genai
from config import GEMINI_MODEL, TEMPERATURE, TOP_K, TOP_P, MAX_OUTPUT_TOKENS
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import TFIDF_MATRIX_FILE, VECTORIZER_FILE, DATA_DIR
import re
from pyvi import ViTokenizer
import joblib
from pathlib import Path
from config import CURRENT_DIR, STOPWORDS_FILE
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,
        host=MYSQL_HOST,
        port=int(MYSQL_PORT),
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
except:
    connection_pool = None

@contextlib.contextmanager
def get_connection():
    conn = None
    try:
        if connection_pool:
            conn = connection_pool.get_connection()
        else:
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                port=int(MYSQL_PORT),
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
        yield conn
    except Error:
        yield None
    finally:
        if conn and (hasattr(conn, 'is_connected') and conn.is_connected()):
            conn.close()

def get_query_questions():
    return """
    SELECT q.id, q.content, q.created_at, q.title, q.status_approval, q.role_ask_id, q.user_id
    FROM question q
    WHERE q.status_delete = 0
    """

def get_query_answers():
    return """
    SELECT a.id, a.content, a.created_at, a.question_id, a.status_answer, a.status_approval,
           a.title, a.role_consultant_id, a.user_id
    FROM answer a
    """

def fetch_data_from_mysql():
    with get_connection() as connection:
        if not connection:
            return pd.DataFrame()

        try:
            questions_df = pd.read_sql(get_query_questions(), connection)
            answers_df = pd.read_sql(get_query_answers(), connection)
            
            if questions_df.empty or answers_df.empty:
                return pd.DataFrame()
                
            merged_df = pd.merge(
                questions_df,
                answers_df,
                left_on='id',
                right_on='question_id',
                how='inner',
                suffixes=('_question', '_answer')
            )
            
            if merged_df.empty:
                merged_df = pd.merge(
                    questions_df,
                    answers_df,
                    left_on='id',
                    right_on='question_id',
                    how='left',
                    suffixes=('_question', '_answer')
                )
                merged_df = merged_df[merged_df['question_id'].notna()]
            
            if merged_df.empty:
                return pd.DataFrame()
            
            result_df = pd.DataFrame({
                'question': merged_df['content_question'],
                'answer': merged_df['content_answer'],
                'question_id': merged_df['id_question'],
                'answer_id': merged_df['id_answer'],
                'source': 'mysql'
            })
            
            return result_df
        
        except Error:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

def prepare_data():
    try:
        mysql_df = fetch_data_from_mysql()
        
        if mysql_df.empty:
            return mysql_df, None, None
        
        df = mysql_df
        
        if df.empty:
            df = pd.DataFrame(columns=['question', 'answer', 'source'])
        
        df['question'] = df['question'].astype(str).fillna('')
        df['answer'] = df['answer'].astype(str).fillna('')
        df = df.drop_duplicates(subset=['question'], keep='last').reset_index(drop=True)
        
        df['question_tokenized'] = df['question'].apply(tokenize_vietnamese)
        df['answer_tokenized'] = df['answer'].apply(tokenize_vietnamese)
        df['content'] = df['question_tokenized'] + ' ' + df['answer_tokenized']
        
        vietnamese_stopwords = load_stopwords()
        vectorizer, tfidf_matrix = create_tfidf_model(df, vietnamese_stopwords)
        
        return df, vectorizer, tfidf_matrix
    except Exception as e:
        return pd.DataFrame(columns=['question', 'answer', 'source']), None, None


def tokenize_vietnamese(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        tokenized = " ".join(ViTokenizer.tokenize(text).split())
        return tokenized
    except Exception:
        return text

def create_tfidf_model(df, stopwords):
    try:
        vectorizer = TfidfVectorizer(
            min_df=2,
            max_features=10000,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            stop_words=stopwords
        )
        content = df['content'] if len(df) > 0 else ["fallback content"]
        tfidf_matrix = vectorizer.fit_transform(content)
        
        try:
            tfidf_path = get_data_path(TFIDF_MATRIX_FILE)
            vectorizer_path = get_data_path(VECTORIZER_FILE)
            joblib.dump(tfidf_matrix, tfidf_path)
            joblib.dump(vectorizer, vectorizer_path)
        except Exception as e:
            print(f"Error saving TF-IDF model: {str(e)}")
        return vectorizer, tfidf_matrix
    except Exception as e:
        return None, None

def load_stopwords():
    try:
        stopwords_path = get_data_path(STOPWORDS_FILE)
        if not stopwords_path.exists():
            return []
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        return stopwords
    except Exception as e:
        return [] 

def get_data_path(filename):
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    current_path = CURRENT_DIR / filename
    if current_path.exists():
        return current_path
    return Path(filename)

def personalize_answer(question, original_answer):
    prompt = f"""
    NHIỆM VỤ: Điều chỉnh câu trả lời gốc để phù hợp với thông tin trong câu hỏi mới.
    
    CÂU HỎI MỚI: {question}
    
    CÂU TRẢ LỜI GỐC: {original_answer}
    
    HƯỚNG DẪN:
    1. Giữ nguyên cấu trúc và ý chính của câu trả lời gốc
    2. Chỉ thay đổi các thông tin cụ thể (số tín chỉ, học kỳ, năm học, số tiền, v.v.) để phù hợp với câu hỏi mới
    3. Đảm bảo câu trả lời vẫn mạch lạc và tự nhiên
    4. Không thêm thông tin mới ngoài những gì có trong câu trả lời gốc
    
    TRẢ LỜI (chỉ trả về câu trả lời đã điều chỉnh, không giải thích):
    """
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                top_p=TOP_P,
                top_k=TOP_K,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        )
        return response.text.strip() if hasattr(response, 'text') else original_answer
    except Exception:
        return original_answer