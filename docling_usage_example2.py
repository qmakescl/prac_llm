# https://docling-project.github.io/docling/examples/custom_convert/

# 설정 및 로깅 초기화
import json, logging, time
from pathlib import Path

_log = logging.getLogger(__name__)


# 핵심도구 Import
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "./data"
    input_doc_path = data_folder / "pdf/RIS-ING-14.pdf" # 한글 파일 연습

    # Pipeline 구성
    pipeline_options = PdfPipelineOptions()
    # pipeline_options.ocr_options.use_gpu = True  # GPU 사용 설정이지만, 설정을 안 할 때 metal 사용, CPU 사용시에만 설정
    # OCR 및 표 구조 분석 기능 활성화
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    # OCR 언어로 한국어, 영어와 스페인어[es] 설정, 한글은 영어하고만 사용 가능
    pipeline_options.ocr_options.lang = ["ko", "en"]
    # CPU 또는 GPU 자동 선택 기능 적용
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )    

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    output_dir = Path("./data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # Export Deep Search document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Export Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_text())

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # Export Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_document_tokens())

if __name__ == "__main__":
    main()
