"""Read the two source worksheets and retain the frozen EC3 luminal protein set."""
import collections
import hashlib
import io
import json
import math
import posixpath
import re
import zipfile
from xml.etree import ElementTree as ET

WORKBOOK = "leitner_2026_S5.xlsx"
TARGETS = {"P07339": "CTSD", "O14773": "TPP1"}
SAMPLE = re.compile(r"(AD|MCI|Control)__(p|np)__(B\d+|ROS\d+)")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def worksheet_rows(book, member, strings):
    with book.open(member) as stream:
        for _, node in ET.iterparse(stream, events=("end",)):
            if node.tag != NS + "row":
                continue
            row = {}
            for cell in node.iterfind(NS + "c"):
                value = cell.findtext(NS + "v")
                if value is None:
                    continue
                if cell.get("t") == "s":
                    value = strings[int(value)]
                key = cell.get("r").rstrip("0123456789")
                row[key] = value
            yield int(node.get("r")), row
            node.clear()
