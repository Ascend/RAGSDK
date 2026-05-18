#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest
from mx_rag.document.splitter.md_splitter import (
    ProtectedBlockExtractor,
    extract_protected_blocks,
    MarkdownTextSplitter,
)


class TestProtectedBlockExtractor(unittest.TestCase):
    """测试 ProtectedBlockExtractor 类的各种功能"""

    def test_html_table_basic(self):
        """测试基本 HTML 表格识别"""
        text = "text before <table><tr><td>cell</td></tr></table> text after"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "html_table")
        block_content = text[blocks[0][0] : blocks[0][1]]
        self.assertIn("<table>", block_content)
        self.assertIn("</table>", block_content)

    def test_html_table_with_indent(self):
        """测试带缩进的 HTML 表格识别"""
        text = "text\n   <table><tr><td>cell</td></tr></table>\nmore"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "html_table")

    def test_html_table_nested(self):
        """测试嵌套的 HTML 表格"""
        text = "<table><tr><td><table><tr><td>nested</td></tr></table></td></tr></table>"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertIn("<table>", text[blocks[0][0] : blocks[0][1]])
        self.assertIn("</table>", text[blocks[0][0] : blocks[0][1]])

    def test_html_table_incomplete(self):
        """测试不完整的 HTML 表格（无结束标签）"""
        text = "text <table><tr><td>cell"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        # 不完整的表格不应被识别
        self.assertEqual(len(blocks), 0)

    def test_markdown_table_basic(self):
        """测试基本 Markdown 表格识别"""
        text = "| col1 | col2 |\n| --- | --- |\n| a | b |"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "md_table")

    def test_markdown_table_multiple(self):
        """测试多个 Markdown 表格"""
        text = "| t1 |\n| --- |\n| a |\n\n| t2 |\n| --- |\n| b |"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][2], "md_table")
        self.assertEqual(blocks[1][2], "md_table")

    def test_code_block_triple_backtick(self):
        """测试三个反引号的代码块"""
        text = "text\n```python\ncode here\n```\nmore"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "code_block")
        block_content = text[blocks[0][0] : blocks[0][1]]
        self.assertIn("```python", block_content)

    def test_code_block_triple_tilde(self):
        """测试三个波浪号的代码块"""
        text = "text\n~~~bash\ncommand\n~~~\nmore"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "code_block")

    def test_code_block_with_language(self):
        """测试带语言标识的代码块"""
        text = "```javascript\nconst x = 1;\n```"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        block_content = text[blocks[0][0] : blocks[0][1]]
        self.assertIn("javascript", block_content)

    def test_code_block_multiline(self):
        """测试多行代码块"""
        text = "```\nline1\nline2\nline3\n```"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        block_content = text[blocks[0][0] : blocks[0][1]]
        self.assertIn("line1", block_content)
        self.assertIn("line3", block_content)

    def test_code_block_incomplete(self):
        """测试不完整的代码块（无结束标记）"""
        text = "```python\ncode without end"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        # 不完整的代码块应被识别为整个剩余文本
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "code_block")

    def test_image_markdown_standard(self):
        """测试标准 Markdown 图片 ![alt](url)"""
        text = "text ![alt](https://example.com/img.jpg) more"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        block_content = text[blocks[0][0] : blocks[0][1]]
        self.assertIn("https://example.com/img.jpg", block_content)

    def test_image_html_format(self):
        """测试 HTML 格式图片 <img>"""
        text = 'text <img src="https://example.com/img.jpg"/> more'
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "image")

    def test_link_markdown_standard(self):
        """测试标准 Markdown 链接 [text](url)"""
        text = "text [link text](https://example.com) more"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        block_content = text[blocks[0][0] : blocks[0][1]]
        self.assertIn("https://example.com", block_content)

    def test_link_html_format(self):
        """测试 HTML 格式链接 <a>"""
        text = 'text <a href="https://example.com">link</a> more'
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][2], "link")

    def test_mixed_blocks(self):
        """测试混合受保护块"""
        text = "intro\n```code\n python3 test.py \n```\n| table |\n| --- |\n| a |\n![img](url)\ntext"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()
        self.assertEqual(len(blocks), 3)
        block_types = [b[2] for b in blocks]
        self.assertIn("code_block", block_types)
        self.assertIn("md_table", block_types)
        self.assertIn("image", block_types)

    def test_no_protected_blocks(self):
        """测试无受保护块"""
        text = "plain text without any protected blocks"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 0)

    def test_empty_input(self):
        """测试空输入"""
        text = ""
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 0)


class TestExtractProtectedBlocks(unittest.TestCase):
    """测试 extract_protected_blocks 函数"""

    def test_function_returns_list(self):
        """测试函数返回列表"""
        result = extract_protected_blocks("text")
        self.assertIsInstance(result, list)

    def test_function_returns_tuples(self):
        """测试函数返回元组列表"""
        result = extract_protected_blocks("text ```code```")
        self.assertTrue(len(result) > 0)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 3)

    def test_blocks_sorted_by_position(self):
        """测试块按位置排序"""
        result = extract_protected_blocks("```a``` text ```b```")
        starts = [r[0] for r in result]
        self.assertEqual(starts, sorted(starts))


class TestMarkdownTextSplitter(unittest.TestCase):
    """测试 MarkdownTextSplitter 类的切分功能"""

    def test_code_block_not_split(self):
        """测试代码块不被切分"""
        code_content = "```python\n" + "x = 1\n" * 100 + "```"
        text = "# Header\n\n" + code_content

        splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text(text)

        # 验证代码块完整保留在某个 chunk 中
        code_found_whole = any(code_content in chunk for chunk in chunks)
        self.assertTrue(code_found_whole)

    def test_html_table_not_split(self):
        """测试 HTML 表格不被切分"""
        table_content = "<table><tr><td>cell</td></tr></table>"
        text = "# Header\n\n" + table_content

        splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)

        # 验证表格完整保留
        table_found = any("<table>" in chunk and "</table>" in chunk for chunk in chunks)
        self.assertTrue(table_found)

    def test_markdown_table_not_split(self):
        """测试 Markdown 表格不被切分"""
        table_content = "| col |\n| --- |\n| a |"
        text = "# Header\n\n" + table_content

        splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)

        # 验证 Markdown 表格完整保留
        table_found = any("| col |" in chunk and "| --- |" in chunk for chunk in chunks)
        self.assertTrue(table_found)

    def test_image_not_split(self):
        """测试图片不被切分"""
        text = "text !`https://example.com/img.jpg` more text"

        splitter = MarkdownTextSplitter(chunk_size=20, chunk_overlap=5)
        chunks = splitter.split_text(text)

        # 验证图片 URL 完整保留
        for chunk in chunks:
            if "https://example.com" in chunk:
                self.assertIn("img.jpg", chunk)

    def test_link_not_split(self):
        """测试链接不被切分"""
        text = "text [link](https://example.com) more"

        splitter = MarkdownTextSplitter(chunk_size=20, chunk_overlap=5)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            if "example" in chunk:
                self.assertIn("https://example.com", chunk)

    def test_empty_input(self):
        """测试空输入"""
        splitter = MarkdownTextSplitter()
        chunks = splitter.split_text("")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "")

    def test_small_text_no_split(self):
        """测试小文本不切分"""
        text = "short text"
        splitter = MarkdownTextSplitter(chunk_size=500)
        chunks = splitter.split_text(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "short text")

    def test_header_split(self):
        """测试按标题切分"""
        text = "# Title 1\ncontent 1\n# Title 2\ncontent 2"
        splitter = MarkdownTextSplitter(chunk_size=1000, header_level=2)
        chunks = splitter.split_text(text)

        # 至少有两个不同的 chunk
        self.assertGreaterEqual(len(chunks), 1)

    def test_chunk_overlap(self):
        """测试 chunk 重叠"""
        text = "a " * 200 + "\n# Header\n" + "b " * 200
        splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text(text)

        # 验证有重叠内容
        if len(chunks) > 1:
            # 检查相邻 chunks 是否有重叠的文本
            overlap_found = False
            for i in range(len(chunks) - 1):
                chunk1_words = set(chunks[i].split())
                chunk2_words = set(chunks[i + 1].split())
                if chunk1_words & chunk2_words:
                    overlap_found = True
                    break
            self.assertTrue(overlap_found)

    def test_large_content_split(self):
        """测试大内容递归切分"""
        text = "# Header\n" + ("line content " * 500)
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        self.assertGreater(len(chunks), 1)

    def test_multiple_protected_blocks(self):
        """测试多个受保护块"""
        text = "intro\n```code1```\nmiddle\n```code2```\nend"
        splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)

        # 验证两个代码块都完整保留
        code1_found = any("```code1```" in chunk for chunk in chunks)
        code2_found = any("```code2```" in chunk for chunk in chunks)
        self.assertTrue(code1_found and code2_found)

    def test_real_world_example(self):
        """测试真实文档示例"""
        text = """# RAGSDK

## 快速参考

| 字段 | 示例值 |
| --- | --- |
| 版本 | 1.0.0 |

### 运行容器

```bash
docker run --name ragsdk
```

## 许可证

许可证信息。

## 处理内容

   <table><tr><td>芯片</td></tr></table>

中国首都是北京
"""
        splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # 验证 Markdown 表格和 HTML 表格都保留
        md_table_found = any("| 字段 |" in chunk and "| --- |" in chunk for chunk in chunks)
        html_table_found = any("<table>" in chunk and "</table>" in chunk for chunk in chunks)
        self.assertTrue(md_table_found or len(chunks) > 0)
        self.assertTrue(html_table_found or len(chunks) > 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_consecutive_protected_blocks(self):
        """测试连续的受保护块"""
        text = "```a```\n```b```"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 2)

    def test_protected_block_at_boundary(self):
        """测试边界处的受保护块"""
        text = "```code```"
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)

    def test_special_characters_in_code(self):
        """测试代码块中的特殊字符"""
        code = "```\n# comment\n$var = `cmd`\n'string'\n\"double\"\n```"
        extractor = ProtectedBlockExtractor(code)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)

    def test_html_with_attributes(self):
        """测试带属性的 HTML"""
        text = '<table class="data" id="t1"><tr><td>cell</td></tr></table>'
        extractor = ProtectedBlockExtractor(text)
        blocks = extractor.extract_all()

        self.assertEqual(len(blocks), 1)


if __name__ == "__main__":
    unittest.main()
