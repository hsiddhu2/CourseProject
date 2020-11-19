import os
import fnmatch
import re
import shutil
import xml.etree.ElementTree as ET

for path, dirs, files in os.walk('NYTData'):
    for file in files:
        if fnmatch.fnmatch(file, '*.xml'):
            fullname = os.path.join(path, file)
            root = ET.parse(fullname).getroot()
            for paragraphs in root.findall('body/body.content/block/p'):
                body_para = paragraphs.text
                if body_para is not None:
                    if re.match(r'\bGore(?!\.?\d)', body_para):
                        print(body_para)
                        shutil.copy(fullname, 'Data/BushGore')
                    if re.match(r'\bBush(?!\.?\d)', body_para):
                        print(body_para)
                        shutil.copy(fullname, 'Data/BushGore')

