from bs4 import BeautifulSoup


def formatXML():
    with open('Data/BushGoreXML.xml') as xmldata:
        prettyXML = BeautifulSoup(xmldata, "xml").prettify()

    with open("Data/BushGorePretty.xml", "w") as f:
        f.write(prettyXML)


def main():

    formatXML()


if __name__ == "__main__":
    main()
