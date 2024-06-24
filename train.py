# Import thư viện
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm 

# Load dữ liệu
file = open('D:/Huan/ProjectII/training_data.txt', 'r', encoding='utf-8')
c_qs = '<qs>'
c_as = '<as>'
data = [i for i in file.read().split(c_qs) if len(i) != 0]

# data = {(question, answer)}
data = [i.split(c_as) for i in data if len(i.split(c_as)) == 2]
file.close()
questions = [i[0] for i in data]
answers = [i[1] for i in data]

# Tiền sử lý câu hỏi

# Loại bỏ khoảng trống thừa
def remove_extra_whitespace(text):
    return " ".join(text.split())

# Ví dụ: chuẩn hóa dấu ngoặc kép
def normalize_special_characters(text):
    
    text = text.replace('“', '"').replace('”', '"')
    return text

def preprocess_text(text):
    text = text.lower()
    text = remove_extra_whitespace(text)
    text = normalize_special_characters(text)
    return text

questions = [preprocess_text(i) for i in questions]

# Tạo embedding cho tập câu hỏi

# Load mô hình embedding đã được huấn luyện trước
model = SentenceTransformer('dangvantuan/vietnamese-embedding')

tokenizer_sent = [tokenize(sent) for sent in questions]
embeddings = model.encode(tokenizer_sent)

# Lưu embeddings của tập câu hỏi để tái sử dụng
np.save('D:/Huan/ProjectII/embeddings.npy', embeddings)

print(embeddings)
