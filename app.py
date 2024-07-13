# Tạo giao diện chatbot
from flask import Flask, render_template, request
from module import Chatbot

bot = Chatbot()

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Route cho trang chủ
@app.route('/')
def index():
    return render_template('index.html') 

# Route để xử lý yêu cầu từ người dùng và trả về câu trả lời từ chatbot
@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message']
    bot_response = bot.get_response(user_message) 
    return bot_response  

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
