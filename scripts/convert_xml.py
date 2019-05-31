import xml.etree.ElementTree as ET
from sys import argv


def main():
    for name in argv[1:]:
        xml_text = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        with open(name) as file:
            xml_text += "".join(file.readlines()).replace('&', '')

        raw_text = ""
        text_root = ET.fromstring(xml_text).findall(".//TEXT")[0]
        if text_root.text.strip() != '':
            raw_text = text_root.text
        else:
            for paragraph in text_root.findall(".//P"):
                raw_text += paragraph.text

        with open(name + ".PRUNED.txt", 'w') as file:
            file.write(raw_text)

        xml_text = ""
        raw_text = ""


if __name__ == "__main__":
    main()

