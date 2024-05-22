from automodeldocs.embed.embed_files import embed_libs
from automodeldocs.process import DescribeCodeBlock


def test_module_explanation():
    embed_libs(["paddleocr"], ["paddleocr", "ppocr"])
    with open(r"D:\PaddleOCR\tools\train.py") as f:
        code = f.read()
    print(DescribeCodeBlock(code).describe_function())


if __name__ == "__main__":
    test_module_explanation()
