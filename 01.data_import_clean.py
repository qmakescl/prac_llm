import re
from pathlib import Path

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