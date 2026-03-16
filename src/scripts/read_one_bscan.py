import pydicom

file_path = "/optima/data/OPTDATAMELBOURNE/105523332-25-IHYAEAVYAEAAAGEAYXYLFAFGA+XBVFUDBSKJWAFVYAFTHUDMVUPETBPZGMIEDITR+TFSRWD/bscan.dcm"

try:
    print(f"Reading: {file_path}\n")
    
    # Read the file header (ignoring the pixel data -> it gives error when loading)
    dcm = pydicom.dcmread(file_path, force=True)
    
    # Print the Transfer Syntax -> Image compression info
    print("--- ENCODING INFO ---")
    if 'TransferSyntaxUID' in dcm.file_meta:
        syntax = dcm.file_meta.TransferSyntaxUID
        print(f"Transfer Syntax UID: {syntax} ({syntax.name})")
    else:
        print("Transfer Syntax UID: MISSING")

    # Check image tags
    print("\n--- IMAGE TAGS ---")
    tags_to_check = [
        ('SamplesPerPixel', '(0028, 0002)'),
        ('Rows', '(0028, 0010)'),
        ('Columns', '(0028, 0011)'),
        ('NumberOfFrames', '(0028, 0008)')
    ]
    
    for name, tag in tags_to_check:
        if name in dcm:
            print(f"{name}: {getattr(dcm, name)}")
        else:
            print(f"{name}: MISSING")

    # Check if Pixel Data actually exists in this file
    print("\n--- PIXEL DATA ---")
    if 'PixelData' in dcm:
        print(f"PixelData: Present (Length: {len(dcm.PixelData)} bytes)")
    else:
        print("PixelData: MISSING")

except Exception as e:
    print(f"Failed to read file: {e}")