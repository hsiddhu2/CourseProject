import os
import fnmatch
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime


def getdate(root):
    day = root.find(".//head/meta[@name='publication_day_of_month']").attrib["content"]
    month = root.find(".//head/meta[@name='publication_month']").attrib["content"]
    year = root.find(".//head/meta[@name='publication_year']").attrib["content"]
    publish_date = month + '-' + day + '-' + year
    return publish_date


def getBushGoreXMLs():
    for path, dirs, files in os.walk('NYTData'):
        for file in files:
            if fnmatch.fnmatch(file, '*.xml'):
                fullname = os.path.join(path, file)
                root = ET.parse(fullname).getroot()
                for paragraphs in root.findall('body/body.content/block/p'):
                    body_para = paragraphs.text
                    if body_para is not None:
                        if re.search(r'\bGore(?!\.?\d) | \bBush(?!\.?\d)', body_para):
                            publish_date = getdate(root)
                            content = publish_date + ': ' + body_para
                            with open("Data/BushGore.txt", "a") as f:
                                f.write(content + "\n")
                            shutil.copy(fullname, 'Data/BushGore')
                        # if re.search(r'\bBush(?!\.?\d)', body_para):
                        #     publish_date = getdate(root)
                        #     content = publish_date + ': ' + body_para
                        #     with open("Data/BushGore.txt", "a") as f:
                        #         f.write(content+ "\n")
                        #     shutil.copy(fullname, 'Data/BushGore')


def main():
    getBushGoreXMLs()


if __name__ == "__main__":
    main()
