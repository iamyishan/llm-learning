"""
1.写文件
2. 读文件
3.追加
4.网络搜索
"""
import os


def _get_workdir_root():
    workdir_root = os.environ.get("WORKDIR_ROOT", "./data/llm_result")
    return workdir_root


WORKDIR_ROOT = _get_workdir_root()


def read_file(filename):
    filename = os.path.join(WORKDIR_ROOT, filename)
    if not os.path.exists(filename):
        return print(f"{filename}文件不存在")
    with open(filename, 'r') as f:
        return "\n".join(f.readline())


def append_to_file(filename, content):
    filename = os.path.join(WORKDIR_ROOT, filename)

    if not os.path.exists(filename):
        return print(f"{filename}文件不存在")

    with open(filename, 'a') as f:
        f.write(content)
    return "append content to file success"

def write_to_file(filename, content):
    filename = os.path.join(WORKDIR_ROOT, filename)
    if not os.path.exists(WORKDIR_ROOT):
        pass
    with open(filename, 'a') as f:
        f.write(content)
    return "append content to file success"
