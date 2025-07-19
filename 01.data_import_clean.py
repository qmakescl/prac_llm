import re
from pathlib import Path

## Data Downloaded from Kaggle - https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books
data_dir = Path('./data/HP')
data_dir.mkdir(parents=True, exist_ok=True)


def clean_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        book_text = file.read()

    cleaned_text = re.sub(r'\n+', ' ', book_text) # 줄바꿈을 빈칸으로 변경
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # 여러 빈칸을 하나의 빈칸으로

    base_filename = filename.name
    print("cleaned_" + base_filename, len(cleaned_text), "characters") # 글자 수 출력

    output_path = data_dir / f"cleaned_{base_filename}"

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# 파일을 찾을 폴더 경로를 Path 객체로 생성
print(f"'{data_dir}' 폴더에서 .txt 파일을 검색합니다...")

# Path 객체의 glob 메서드를 사용하여 .txt 파일을 찾습니다.
# 결과는 제너레이터(generator)이므로 list()로 변환해줍니다.
try:
    filenames_list = list(data_dir.glob('*.txt'))
    filenames_list = [file for file in filenames_list if not file.name.startswith("cleaned_")]

    # 결과 출력
    if filenames_list:
        print("\n[성공] 발견된 .txt 파일 목록:")
        for file_path in filenames_list:
            # Path 객체는 그대로 출력해도 경로가 잘 보입니다.
            # 문자열이 필요하면 str(file_path)로 변환할 수 있습니다.
            print(file_path)
    else:
        print(f"\n[알림] 해당 경로에 .txt 파일이 없습니다: {data_dir}")

except FileNotFoundError:
    print(f"\n[오류] 폴더를 찾을 수 없습니다: {data_dir}")

# filenames_list = ["02 Harry Potter and the Chamber of Secrets.txt"]

for filename in filenames_list:
    clean_text(filename)

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hi! How are you? I am fine. Thank you."
tokens = tokenizer.encode(text)

print("글자수:", len(text), "토큰수", len(tokens))
print(tokens)
print(tokenizer.decode(tokens))
for t in tokens:
    print(f"{t}\t -> {tokenizer.decode([t])}")


# 한글 예제
from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")

text = "안녕하세요! 잘 지내고 계신가요? 저는 잘 지내고 있습니다. 감사합니다."

tokens = tokenizer.encode(text)

print("글자수:", len(text), "토큰수", len(tokens))
print(tokens)
print(tokenizer.decode(tokens))
for t in tokens:
    print(f"{t}\t -> {tokenizer.decode([t])}")


# DataLoader
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # token_ids = tokenizer.encode("<|endoftext|>" + txt, allowed_special={"<|endoftext|>"})
        token_ids = tokenizer.encode(txt)

        print("# of tokens in txt:", len(token_ids))

        # input 과 target 생성 
        for i in range(0, len(token_ids) - max_length, stride):
            # target 에 는 다음 토큰을 포함하기 위해 i+1 -> 훈련시 바로 다음 단어를 예측하도록 하기 위해
            # input_chunk 는 max_length 길이로 자르고, target_chunk 는 input_chunk 다음 토큰부터 시작
            # stride 만큼 이동하면서 반복
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
# with open("cleaned_한글문서.txt", 'r', encoding='utf-8-sig') as file: # 선택: -sig를 붙여서 BOM 제거
# with open("./data/HP/cleaned_02 Harry Potter and the Chamber of Secrets.txt", 'r', encoding='utf-8-sig') as file: # 선택: -sig를 붙여서 BOM 제거
#     txt = file.read()


cleaned_txt_files = sorted(data_dir.glob('cleaned_*.txt'))
all_texts = ""

for file_path in cleaned_txt_files:
    print(f"읽는 중: {file_path.name}")
    try:
        # pathlib의 .read_text()를 사용하면 파일을 열고, 읽고, 닫는 과정을 한번에 처리합니다.
        # 'utf-8-sig' 인코딩은 파일 시작 부분의 보이지 않는 BOM(Byte Order Mark)을 자동으로 처리해줍니다.
        content = file_path.read_text(encoding='utf-8-sig')
        all_texts += content + " "
    except FileNotFoundError:
        print(f"[경고] 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"[오류] 파일을 읽는 중 문제가 발생했습니다: {file_path}, {e}")

dataset = MyDataset(all_texts, max_length = 32, stride = 4)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)


# Dataloader 로 만든 내용 확인
# output이 하나씩 밀려있음
dataiter = iter(train_loader)

x, y = next(dataiter)

# input / output 한칸씩 밀린 것 확인
print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))

