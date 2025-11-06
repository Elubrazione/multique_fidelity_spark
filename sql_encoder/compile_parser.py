# 创建一个编译脚本 compile_parser.py
from tree_sitter import Language

# 编译 SQL 语法解析器
Language.build_library(
    'sql_encoder/tree-sitter/sql.dll',  # Windows 用 .dll
    ['tree-sitter-sql']  # 这会自动从 GitHub 下载
)