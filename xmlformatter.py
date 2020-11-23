from operator import attrgetter

from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


def formatXML():
    with open('Data/BushGore2.xml') as xmldata:
        prettyXML = BeautifulSoup(xmldata, "xml").prettify()

    with open("Data/BushGorePretty.xml", "w") as f:
        f.write(prettyXML)


def getKeyValue(elem):
    return elem.findtext("number")


def sortXML():
    tree = ET.parse('Data/BushGorePretty.xml')
    container = tree.find("entries")

    data = []
    for elem in container:
        key = elem.findtext("date")
        key = int(key)
        data.append((key, elem))

    data.sort(key=lambda t: t[0])

    # insert the last item from each tuple
    container[:] = [item[-1] for item in data]

    tree.write('Data/SortedXML.xml')


def main():
    formatXML()
    sortXML()


if __name__ == "__main__":
    main()
