import os
import fnmatch
import re
import shutil
import xml
import xml.etree.ElementTree as ET
import xmlformatter


def getdate(root):
    day = root.find(".//head/meta[@name='publication_day_of_month']").attrib["content"]
    month = root.find(".//head/meta[@name='publication_month']").attrib["content"]
    year = root.find(".//head/meta[@name='publication_year']").attrib["content"]
    publish_date = month + day + year
    return publish_date


def writeXML(publish_date, body_para):
    output_XML = ET.parse("Data/BushGore2.xml")
    rootNode = output_XML.getroot()
    entries = output_XML.find("entries")

    entry = ET.SubElement(entries, "entry")
    date = ET.SubElement(entry, "date")
    textdata = ET.SubElement(entry, "textdata")

    date.text = publish_date
    textdata.text = body_para

    tree = ET.ElementTree(rootNode)
    tree.write("Data/BushGore2.xml", "utf-8")


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
                            # shutil.copy(fullname, 'Data/BushGore')
                            # writeXML(publish_date, body_para)


def createOutputXML():
    root = ET.Element("presidential")
    ET.SubElement(root, "entries")
    tree = ET.ElementTree(root)

    tree.write("Data/BushGore2.xml")


def refreshTextFile():
    open( 'Data/BushGore.txt', 'w' ).close()


def main():
    refreshTextFile()
    createOutputXML()
    getBushGoreXMLs()


if __name__ == "__main__":
    main()
