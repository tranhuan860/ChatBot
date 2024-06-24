# Tạo giao diện chatbot
from flask import Flask, render_template, request
from module import Chatbot
import re
from pyvi.ViTokenizer import tokenize

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

bot = Chatbot()

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Route cho trang chủ
@app.route('/')
def index():
    return render_template('index.html')  # Trả về template HTML cho trang chủ

# Route để xử lý yêu cầu từ người dùng và trả về câu trả lời từ chatbot
@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message']  # Lấy thông điệp từ người dùng
    bot_response = bot.get_response(tokenize(preprocess_text(user_message)))  # Gọi hàm để lấy câu trả lời từ chatbot
    return bot_response  # Trả về câu trả lời cho người dùng

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
