from automodeldocs.embed.embed_files import embed_libs
from automodeldocs.process import DescribeFunction


def test_function_explain():
    embed_libs(["tabpfn"], ["tabpfn"])
    print(DescribeFunction("TabPFNClassifier").describe_function())


if __name__ == "__main__":
    test_function_explain()
