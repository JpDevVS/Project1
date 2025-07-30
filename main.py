import os
import shutil
import xml.etree.ElementTree as ET
import re


def load_xml_to_string(xml_file_path):
    """
    Load XML content from a file into a string.

    Args:
        xml_file_path (str): Path to the XML file

    Returns:
        str: XML content as a string, or None if there was an error
    """
    try:
        with open(xml_file_path, 'r', encoding='utf-8', errors='replace') as file:
            xml_string = file.read()
        return xml_string
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return None


def sort_files_based_on_xml_string(source_directory, xml_content):
    """
    Read file names from an XML string and move files from the source directory
    to either 'found' or 'notfound' folders based on whether they match names in the XML.

    Args:
        source_directory (str): Path to the directory containing files to search through
        xml_content (str): String containing XML with target filenames
    """
    # Create 'found' and 'notfound' directories if they don't exist
    found_dir = os.path.join(os.getcwd(), "C:\\temp\\found")
    notfound_dir = os.path.join(os.getcwd(), "C:\\temp\\notfound")

    for directory in [found_dir, notfound_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    # Make sure source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist.")
        return

    # Parse XML string to get target filenames
    try:
        # Clean the XML content by replacing problematic entities
        cleaned_xml = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)[^;]+;', '_', xml_content)

        # Parse the cleaned XML content
        root = ET.fromstring(cleaned_xml)

        # Extract filenames from XML
        target_files = []
        for file_element in root.findall('.//file'):
            filename = file_element.get('name') or file_element.text
            if filename:
                target_files.append(filename.strip())

        print(f"Loaded {len(target_files)} target filenames from XML: {target_files}")

        # If no filenames found in XML
        if not target_files:
            print("No filenames found in the XML content.")
            #return

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        print("Try checking your XML string for invalid characters or undefined entities.")
        return
    except Exception as e:
        print(f"Error processing XML content: {e}")
        return

    # Get list of files in the source directory
    try:
        files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

        if not files:
            print(f"No files found in {source_directory}")
            return

        # Process each file
        for filename in files:
            source_path = os.path.join(source_directory, filename)

            # Determine destination based on whether filename is in target list
            #if filename in target_files:
            filename_temp = "Reportes" + "\\" + filename
            print(filename_temp)
            print(filename_temp.upper())
            print(filename_temp.lower())

            if filename_temp == "MAEMONEDA.rpt": print("found --->>> MAEMONEDA.rpt")
            if filename_temp in xml_content or filename_temp.lower() in xml_content or filename_temp.upper() in xml_content:
                destination = os.path.join(found_dir, filename)
                result = "found"
            else:
                destination = os.path.join(notfound_dir, filename)
                result = "not found"

            # Move the file
            shutil.move(source_path, destination)
            print(f"Moved '{filename}' to {result} folder")

    except Exception as e:
        print(f"Error processing files: {e}")


# Example usage
if __name__ == "__main__":
    # Source directory containing files to process
    source_dir = "C:\\temp\\reportes\\"

    # Path to the XML file
    xml_file_path = "C:\\temp\\total_24042025.xml"

    # Load XML content from file into a string variable
    xml_string = load_xml_to_string(xml_file_path)

    if xml_string:
        print(f"Processing files in '{source_dir}' based on XML content from '{xml_file_path}'")
        sort_files_based_on_xml_string(source_dir, xml_string)
    else:
        print("Failed to load XML content. Check the file path and try again.")


