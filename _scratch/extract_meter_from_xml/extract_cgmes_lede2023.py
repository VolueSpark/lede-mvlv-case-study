import xml.etree.ElementTree as ET
import polars as pl
import json, os

PATH = os.path.dirname(os.path.abspath(__file__))

def extract_cim_name_info(xml_content):
    """
    Extract mrid and meter_id from the given XML content.

    Args:
    xml_content (str): A string containing the XML content.

    Returns:
    list of dict: A list of dictionaries, each containing 'mrid' and 'meter_id' keys.
    """
    root = ET.fromstring(xml_content)
    cim_namespace = {'cim': "http://ucaiug.org/ns/CIM#"}  # Adjust namespace if necessary

    extracted_info = []

    for name_element in root.findall('.//cim:UsagePoint', cim_namespace):
        name_element_name = name_element.find('cim:IdentifiedObject.mRID', cim_namespace)

        if name_element_name is not None:
            extracted_info.append({
                'mrid': name_element_name.text,  # Strip leading underscore from ID
            })

    return extracted_info

# Example usage:
with open(os.path.join(PATH, 'data/Lede2023/Lede-30-nextgenvpp-MyreneFeeder1_CU.xml')) as fp:
    xml_content = fp.read()


df = pl.from_dicts( extract_cim_name_info(xml_content))
df.write_parquet(os.path.join(PATH, 'output/cgmes_lede2023'))

print(f"We have {df.n_unique('mrid')} unique meter id's with mrid in the {os.path.join(PATH, 'data/Lede2023/Lede-30-nextgenvpp-MyreneFeeder1_CU.xml')} CGMES standard")
