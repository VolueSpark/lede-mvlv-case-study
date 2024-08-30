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
    cim_namespace = {'cim': "http://iec.ch/TC57/2013/CIM-schema-cim16#"}  # Adjust namespace if necessary

    extracted_info = []

    for name_element in root.findall('.//cim:Name', cim_namespace):
        mrid = name_element.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID')
        name_element_name = name_element.find('cim:Name.name', cim_namespace)
        identified_object = name_element.find('cim:Name.IdentifiedObject', cim_namespace)

        if mrid and name_element_name is not None:
            resource = identified_object.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            extracted_info.append({
                'mrid': resource.strip('#_'),  # Strip leading underscore from ID
                'meter_id': name_element_name.text
            })

    return extracted_info

# Example usage:
with open(os.path.join(PATH, 'data/Lede/DIGIN10-24-LV1-P420_EQ.xml')) as fp:
    xml_content = fp.read()


df = pl.from_dicts( extract_cim_name_info(xml_content))
df.write_parquet(os.path.join(PATH, 'output/cgmes'))

print(f"We have {df.n_unique('mrid')} unique meter id's with mrid in the {os.path.join(PATH, 'data/Lede/DIGIN10-24-LV1-P420_EQ.xml')} CGMES standard")
