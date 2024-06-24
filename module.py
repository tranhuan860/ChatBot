from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from pyvi.ViTokenizer import tokenize

class Chatbot:
    def __init__(self):
        # Load dữ liệu
        file = open('D:/Huan/ProjectII/training_data.txt', 'r', encoding='utf-8')
        c_qs = '<qs>'
        c_as = '<as>'
        data = [i for i in file.read().split(c_qs) if len(i) != 0]
        # data = {(question, answer)}
        data = [i.split(c_as) for i in data if len(i.split(c_as)) == 2]
        file.close()
        self.questions = [i[0] for i in data]
        self.answers = [i[1] for i in data]
        self.model = SentenceTransformer('dangvantuan/vietnamese-embedding')
        self.passage_embedding = np.load('D:/Huan/ProjectII/embeddings.npy')
        self.ratio = 0.7
        
    # Trả về câu trả lời và độ tương đồng của câu hỏi với cơ sở dữ liệu
    def get_response(self, question):
        list_questions = question.split('.')
        response = ''
        for qs in list_questions:
            query_embedding = self.model.encode(qs)
            score = util.cos_sim(query_embedding, self.passage_embedding)
            pos = score.argmax()
            if score[0][pos] > self.ratio: 
                response += '\n\n' + self.answers[pos]
        if response == '':
            return 'Xin lỗi tôi không hiểu câu hỏi của bạn'
        return response[2:]
