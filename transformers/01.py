import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. 데이터 준비
sentence = "the cute cat plays with the yarn".split()
vocab = list(set(sentence))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
embedding_dim = 4 # 임베딩 차원: 4

# 2. 학습 데이터 생성 (Skip-gram)
# window_size: 중심 단어로부터 좌우 몇 개까지의 단어를 주변 단어로 볼 것인지 결정
window_size = 2
skip_grams = []
for i in range(len(sentence)):
    center_word = word_to_idx[sentence[i]]
    for j in range(i - window_size, i + window_size + 1):
        if i != j and j >= 0 and j < len(sentence):
            context_word = word_to_idx[sentence[j]]
            skip_grams.append([center_word, context_word])
            # print("Skip-gram pairs:", skip_grams)

# 3. 모델 구축 (PyTorch)
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # 이 부분이 바로 임베딩 층입니다.
        # vocab_size x embedding_dim 크기의 가중치 행렬을 생성합니다.
        # 이 가중치 행렬의 각 행이 특정 단어의 임베딩 벡터가 됩니다.
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # 입력으로 들어온 단어 인덱스에 해당하는 임베딩 벡터를 가져옵니다.
        embedded_vector = self.embeddings(x)
        # 임베딩 벡터를 통해 주변 단어를 예측합니다.
        output = self.linear(embedded_vector)
        return output

# 4. 학습
model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    total_loss = 0
    for center_word, context_word in skip_grams:
        center_tensor = torch.LongTensor([center_word])
        
        # 모델 예측
        pred = model(center_tensor)
        
        # 손실 계산 및 역전파
        loss = criterion(pred, torch.LongTensor([context_word]))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch+1}, Loss: {total_loss:.4f}')

# 학습된 임베딩 벡터 확인
trained_embeddings = model.embeddings.weight.data.numpy()
for word, i in word_to_idx.items():
    print(f'"{word}"의 임베딩 벡터: {trained_embeddings[i]}')


# t-SNE 모델로 2차원으로 차원 축소
tsne_model = TSNE(n_components=2, perplexity=3, random_state=0)
tsne_vectors = tsne_model.fit_transform(trained_embeddings)

# 시각화
plt.figure(figsize=(8, 8))
plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c='steelblue', s=100)

for i, word in enumerate(vocab):
    plt.annotate(word, 
                 xy=(tsne_vectors[i, 0], tsne_vectors[i, 1]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.title('Word Embedding Visualization')
plt.grid(True)
plt.show()