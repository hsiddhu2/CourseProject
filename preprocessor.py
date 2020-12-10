import os
import fnmatch
import re
import xml.etree.ElementTree as ET


"""
Get the publish date from the blog article and format it in MM/DD/YY format.
"""
def getdate(root):
    day = root.find(".//head/meta[@name='publication_day_of_month']").attrib["content"]
    month = root.find(".//head/meta[@name='publication_month']").attrib["content"]
    year = root.find(".//head/meta[@name='publication_year']").attrib["content"]
    publish_date = month.zfill(2)+'/'+day.zfill(2)+'/'+'00'
    return publish_date

"""
Read ALL paragraphs from the XML file and search
keyword "Bush" Or "Gore", if found then read the publish date 
and content and write that to BushGore.txt file in append mode as a new document. 
"""
def readBlockParagraphs(fullname, root):
    for block in root.findall( 'body/body.content/block' ):
        block_type = block.attrib['class']
        if block_type == 'full_text':
            for para in block.findall( 'p' ):
                body_para = para.text
                if body_para is not None:
                    if re.search( r'\bGore(?!\.?\d) | \bBush(?!\.?\d) | \bBush,(?!\.?\d) | \bGore,(?!\.?\d)',
                                  body_para ):
                        publish_date = getdate( root )
                        content = str( publish_date ) + ': ' + body_para
                        with open( "Data/BushGore.txt", "a" ) as f:
                            f.write( content + "\n" )


"""
Read the abstract from the XML file and search
keyword "Bush" Or "Gore", if found then read the publish date 
and content and write that to BushGore.txt file in append mode as a new document. 
"""
def readAbstract(root):
    for paras in root.findall( 'body/body.head/abstract' ):
        for para in paras.findall( 'p' ):
            abstract = para.text
            if abstract is not None:
                if re.search( r'\bGore(?!\.?\d) | \bBush(?!\.?\d) | \bBush,(?!\.?\d) | \bGore,(?!\.?\d)',abstract ):
                    publish_date = getdate( root )
                    content = str( publish_date ) + ': ' + abstract
                    with open( "Data/BushGore.txt", "a" ) as f:
                        f.write( content + "\n" )


"""
Read ALL XML files from nytdata folder
"""
def getBushGoreXMLs():
    for path, dirs, files in os.walk( 'nytdata' ):
        for file in files:
            if fnmatch.fnmatch(file, '*.xml'):
                fullname = os.path.join(path, file)
                root = ET.parse(fullname).getroot()

                """
                Read Blog Abstract and look for keyword 'Bush' Or 'Gore'
                """
                readAbstract(root)

                """
                Read Blog Paragraphs and look for keyword 'Bush' Or 'Gore'
                """
                readBlockParagraphs(fullname, root)


def refreshTextFile():
    open( 'Data/BushGore.txt', 'w' ).close()


"""
Main Method - New York Times Data Extract 
For Year 2000 From 1st May 2000 to 30th October 2000
"""
def main():

    """
    Refresh file to delete the old content if any
    """
    refreshTextFile()

    """ 
    Read all blog articles from nytdata from 1st May 2000
    to 30th Oct 2020 for keyword 'Bush' Or 'Gore'
    """
    getBushGoreXMLs()


if __name__ == "__main__":
    main()
