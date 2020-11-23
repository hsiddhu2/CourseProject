import xml.etree.ElementTree as ET

tree = ET.parse("Data/BushGore2.xml")

container = tree.find("entries")

data = []
for elem in container:
    key = elem.findtext("date")
    data.append((key, elem))

data.sort(key=lambda t: t[0])

print(data)

# insert the last item from each tuple
container[:] = [item[-1] for item in data]

tree.write("Data/test/SampleOutput.xml")
