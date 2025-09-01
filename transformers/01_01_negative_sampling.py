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

# 네거티브 샘플링을 위한 데이터 생성 함수
def create_negative_samples(skip_grams, vocab_size, k):
    inputs, targets, labels = [], [], []
    for center_word, context_word in skip_grams:
        # Positive sample
        inputs.append(center_word)
        targets.append(context_word)
        labels.append(1) # 정답 레이블은 1

        # Negative samples
        for _ in range(k):
            negative_sample = np.random.randint(0, vocab_size)
            # 우연히 정답(context_word)이 뽑히는 것을 방지
            while negative_sample == context_word:
                negative_sample = np.random.randint(0, vocab_size)
            inputs.append(center_word)
            targets.append(negative_sample)
            labels.append(0) # 오답 레이블은 0
    return torch.LongTensor(inputs), torch.LongTensor(targets), torch.FloatTensor(labels)


# 3. 네거티브 샘플링을 적용한 모델 구축
class Word2VecNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecNegativeSampling, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_word, context_word):
        # 각 단어의 임베딩 벡터를 가져옵니다.
        center_embed = self.center_embeddings(center_word)
        context_embed = self.context_embeddings(context_word)
        
        # 두 벡터의 내적(dot product)을 계산합니다.
        # (batch_size, embedding_dim) -> (batch_size, 1) 로 차원 유지
        dot_product = torch.sum(center_embed * context_embed, dim=1)
        
        # Sigmoid 함수를 통해 0~1 사이의 값 (정답일 확률)으로 변환
        return torch.sigmoid(dot_product)

# 4. 학습
embedding_dim = 4
num_negative_samples = 3 # 오답 샘플 개수
inputs, targets, labels = create_negative_samples(skip_grams, vocab_size, num_negative_samples)

model = Word2VecNegativeSampling(vocab_size, embedding_dim)
criterion = nn.BCELoss() # Binary Cross Entropy Loss 사용 (0 또는 1을 예측하므로)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5000): # Epochs 늘리기
    # 모델 예측
    pred = model(inputs, targets)
    
    # 손실 계산 및 역전파
    loss = criterion(pred, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# 학습된 임베딩 벡터는 center_embeddings에 저장되어 있습니다.
trained_embeddings = model.center_embeddings.weight.data.numpy()
print("\n--- 학습 완료된 임베딩 벡터 ---")
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