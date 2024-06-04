# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import csv
from datetime import datetime, timedelta
from typing import List
from loguru import logger
from openpyxl import load_workbook
import xlrd

from mx_rag.document.loader.base_loader import BaseLoader
from mx_rag.document.doc import Doc
from mx_rag.utils import file_check

OPENPYXL_EXTENSION = (".xlsx",)
XLRD_EXTENSION = (".xls",)
CSV_EXTENSION = (".csv",)


class ExcelLoader(BaseLoader):
    def __init__(self, file_path, line_sep="**;"):
        super().__init__(file_path)
        self.line_sep = str(line_sep)

    @staticmethod
    def _exceltime_to_datetime(exceltime):
        """
        将excel储存的时间格式转换为可读的格式
        :param exceltime: (0,1)区间的浮点数，表示时间在一天中的位置
        :return: 以 时：分 的格式返回字符串
        """
        base_date = datetime(1899, 12, 30)
        time = base_date + timedelta(exceltime)
        return time.strftime("%H:%M")

    @staticmethod
    def _cleanup_xlsx(worksheet):
        """
        ：返回：去掉左边与上边空白行列后的表单，左上角对其, 需要感知空cell
        """
        # 查找表格最左上角的cell位置
        start_row = 0
        start_col = 0
        for row in worksheet.iter_rows(values_only=True):
            if all(cell is None for cell in row):
                start_row += 1
        for col in worksheet.iter_cols(values_only=True):
            if all(cell is None for cell in col):
                start_col += 1
        # 删除空行和空列
        if start_row:
            worksheet.delete_rows(1, amount=start_row)
        if start_col:
            worksheet.delete_cols(1, amount=start_col)
        return worksheet

    @staticmethod
    def _table_location_xls(worksheet):
        """
        查找表格最左上角的cell位置
        """
        start_row = 0
        start_col = 0
        for i in range(worksheet.nrows):
            row = worksheet.row_values(i)
            if not sum(cell != "" for cell in row):
                start_row += 1
        for i in range(worksheet.ncols):
            col = worksheet.col_values(i)
            if not sum(cell != "" for cell in col):
                start_col += 1
        return start_row, start_col

    @staticmethod
    def _load_csv_line(row, headers):
        text_line = ""
        for ind, ti in enumerate(headers):
            if not str(ti):
                ti = "None"
            if not str(row[ind]):
                row[ind] = "None"
            text_line += str(ti) + ":" + str(row[ind]) + ";"
        return text_line

    def load(self):
        """
        ：返回：逐行读取表,返回 string list
        """
        try:
            file_check.excel_file_check(self.file_path, self.MAX_SIZE_MB)
        except Exception as e:
            logger.error(e)
            return []
        # 判断文件种类：支持 xlsx 与 xls 格式
        if self.file_path.endswith(XLRD_EXTENSION):
            return self._load_xls()
        elif self.file_path.endswith(OPENPYXL_EXTENSION):
            return self._load_xlsx()
        elif self.file_path.endswith(CSV_EXTENSION):
            return self._load_csv()
        else:
            raise TypeError(f"{self.file_path} file type is not correct")

    def _load_xls_sheet(self, ws):
        res = ""
        start_row, start_col = self._table_location_xls(ws)
        if ws.nrows - start_row < 2:
            logger.info(f"In file {self.file_path} sheet *{ws.name}* is empty")
            return res
        title = ws.row_values(start_row)[start_col:]  # 默认第一排为列名称
        for line_ind in range(start_row + 1, ws.nrows):
            text_line = ""
            line_list = ws.row_values(line_ind)
            for ind, ti in enumerate(title):
                if not str(ti):
                    ti = "None"
                ind += start_col
                if not str(line_list[ind]):
                    line_list[ind] = "None"
                if ti in ["time"] and 0 <= float(line_list[ind]) <= 1:
                    text_line += str(ti) + ":" + str(self._exceltime_to_datetime(float(line_list[ind]))) + ";"
                else:
                    text_line += str(ti) + ":" + str(line_list[ind]) + ";"
            text_line += self.line_sep
            res += text_line
        return res

    def _load_xls(self):
        docs: List[Doc] = list()
        wb = xlrd.open_workbook(self.file_path)
        if wb.nsheets > self.MAX_PAGE_NUM:
            logger.error(f"file {self.file_path} sheets number more than limit")
            return docs
        for i in range(wb.nsheets):  # 对于每一张表
            content = ""
            ws = wb.sheet_by_index(i)
            ws_texts = self._load_xls_sheet(ws)
            content += ws_texts
            doc = Doc(page_content=content, metadata={"source": self.file_path, "sheet": ws.name})
            docs.append(doc)
        logger.info(f"file {self.file_path} Loading completed")
        return docs

    def _load_xlsx(self):
        docs: List[Doc] = list()
        wb = load_workbook(self.file_path)
        if len(wb.sheetnames) > self.MAX_PAGE_NUM:
            logger.error(f"file {self.file_path} sheets number more than limit")
            return docs
        for sheet_name in wb.sheetnames:  # 每张表单
            content = ""
            ws_init = wb[sheet_name]
            ws = self._cleanup_xlsx(ws_init)
            rows = list(ws.rows)
            # 判断表单是否有标题+内容，默认至少两行
            if len(rows) < 2:
                logger.info(f"In file {self.file_path} sheet *{sheet_name}* is empty")
                continue
            title = list(rows[0])
            for line in list(rows[1:]):
                text_line = ""
                for ind, ti in enumerate(title):
                    text_line += str(ti.value) + ":" + str(line[ind].value) + ";"
                content += text_line + self.line_sep
            doc = Doc(page_content=content, metadata={"source": self.file_path, "sheet": sheet_name})
            docs.append(doc)
        logger.info(f"file {self.file_path} Loading completed")
        return docs

    def _load_csv_lines(self, reader, headers):
        content = ""
        for row in reader:
            if len(row) > 0:
                text_line = self._load_csv_line(row, headers)
                content += text_line + self.line_sep
            else:
                break
        return content

    def _load_csv(self):
        docs: List[Doc] = list()
        content = ""
        try:
            with open(self.file_path, mode="r", encoding="utf-8-sig") as file:
                reader = csv.reader(file)
                headers = next(reader)  # 读取第一行标题
                content = self._load_csv_lines(reader, headers)
        except Exception as e:
            raise e
        if content:
            doc = Doc(page_content=content, metadata={"source": self.file_path})
            docs.append(doc)
        else:
            logger.info(f"file {self.file_path} is empty")
        return docs
