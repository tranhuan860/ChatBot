import os
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from g4f.client import Client

G4F_PROXY="http://host:port" 

class Chatbot:
    def __init__(self):
        # Load các biến môi trường từ tệp .env
        load_dotenv(r'E:\Project\Project 2\enviroment.env')

        # Lấy giá trị của các biến môi trường
        QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
        QDRANT_URL = os.getenv('QDRANT_URL')
        EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
        QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')
        
        # Load mô hình embedding
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Kết nối Qdrant
        Qclient = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY, prefer_grpc=False)
        self.db = Qdrant(client=Qclient,
                    embeddings=embeddings,
                    collection_name=QDRANT_COLLECTION_NAME)
        
        self.client = Client()
        self.messages = []
        
    # Trả về câu trả lời và độ tương đồng của câu hỏi với cơ sở dữ liệu
    def get_response(self, question):
        context = '\n'.join([i.page_content for i in self.db.similarity_search(question,k=10)])
        
        # Tạo câu trả lời
        if len(self.messages)==0:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Bạn là một chatbot thông minh có khả năng trả lời câu hỏi dựa trên ngữ cảnh, khi không tìm thấy thông tin hãy trả lời là: Xin lỗi tôi không có thông tin về câu hỏi của bạn, bạn có thể cung cấp cho tôi thêm thông tin về câu hỏi của bạn được không?"},
                    {"role": "user", "content": context},
                    {"role": "user", "content": question}
                ],
                max_tokens=1000
            )
        else:
            ms = [{"role": "system", "content": "Bạn là một chatbot thông minh có khả năng trả lời câu hỏi dựa trên ngữ cảnh, khi không tìm thấy thông tin hãy trả lời là: Xin lỗi tôi không có thông tin về câu hỏi của bạn, bạn có thể cung cấp cho tôi thêm thông tin về câu hỏi của bạn được không?"},]
            for mss in self.messages:
                ms.append({"role": "user", "content": mss[0]})
                ms.append({"role": "assistant", "content": mss[1]})
            ms.append({"role": "user", "content": context})
            ms.append({"role": "user", "content": question})
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=ms,
                max_tokens=1000
            )
        self.messages.append((question, response.choices[-1].message.content))
        if(len(self.messages)>5):
            self.messages = self.messages[1:]
        return response.choices[-1].message.content
