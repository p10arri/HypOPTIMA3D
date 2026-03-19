import pydicom
import numpy as np
from loguru import logger

# Configure logger
logger.remove()  # remove default stdout
logger.add("scripts/read_bscan_sample.log", level="INFO", format="{time} | {level} | {message}")

file_paths = [
    "/optima/data/OPTDATAMELBOURNE/105523332-25-IHYAEAVYAEAAAGEAYXYLFAFGA+XBVFUDBSKJWAFVYAFTHUDMVUPETBPZGMIEDITR+TFSRWD/bscan.dcm",
    "/optima/data/OPTDATA_VSC_DryAMD2/98540702-25-AXAMFIREQPNFOFNCXLYAAVYMA+FCBKOMQHVTNUDKBTYHODNLCKRRAEKCXWRKHDL+LQULKV/bscan.dcm",
    "/optima/data/OPTDATA_FILLY/4204246-25-ZGVMHCORKGSVINHXPVYNEPEBQ+AVOQOCQMGSZATIBXHNAULSGOWTKBAGOSQGMG+VDVSHU/bscan.dcm",
    "/optima/data/OPTDATAVIVID/51871876-25-APJFSFACSVADAAAYMEHAGGAYC+BZKPZPHQLGPXBPYBJNPTZOXYBSNZPEYSLQVTN+STUSEV/bscan.dcm",
    "/optima/data/OPTDATA12/Scans/Spectralis/51871876-25-ADAGGFAOSAAKVDAMYFJAFGAAA+BRESZTRSHARRFDFMNJNVDDZXWBNJDRWREYLNV+HPVDHI/bscan.dcm",
    "/optima/exchange/NORMATIVE_AGING/PatEyeVisit/0289_OS/0720/OrigScan/bscan.dcm",
    "/optima/data/OPTDATASTARGARDTDISTROPHY/26908804-25-AQCAKCDDABQVVALABAKEJAADA+YDNYMKMJRKJZJDXUDNUSNIHVMNYBCAXBEADWN+PLNMLC/bscan.dcm",
    "/optima/data/OPTDATA2/205449860-25-KXVXATAZAQDJIAVYGEYGVCAXQ+YEBWOQURKJGOBXEJAEEEMJRQWENPXLLXVJPTYS+NHCOCR/bscan.dcm",
    "/optima/data/OPTDATA3/4204024-25-FEBHLUZOJCHTKLHNOSDWTZTVT+AKCUIJECATVNBOQOJXDJHAQTWMAFNIRTYSWU+JDYPNY/bscan.dcm",
    "/optima/data/OPTDATA12/Scans/Spectralis/51871876-25-AZDEJAALEAAQVDAMYEIAJGAAD+BGAKXGXZXPNBBMIZAVPVBXPYOJNLZSVEUQNTO+IALFTI/bscan.dcm",
    "/optima/data/OPTDATAHARBOR/B/6782494-25-OSRTVYBLEHLMNWWDJLKIFESUJ+CJALSMAECHRKTEYLDBEDQZXLTJQDENBAFALH+OFXTRA/bscan.dcm"
]


def extract_spacing(dcm):
    """Extract spacing robustly (z, y, x) in mm"""

    #  Z spacing (inter-slice) 
    z = None
    if (0x0018, 0x0088) in dcm:
        try:
            z = float(dcm[(0x0018, 0x0088)].value)
        except:
            pass

    # XY spacing (intra-slice) 
    y, x = None, None
    if (0x0028, 0x0030) in dcm:
        try:
            y, x = map(float, dcm[(0x0028, 0x0030)].value)
        except:
            pass

    return z, y, x


def extract_basic_info(dcm):
    return {
        "Modality": getattr(dcm, "Modality", "MISSING"),
        "Manufacturer": getattr(dcm, "Manufacturer", "MISSING"),
        "Rows": getattr(dcm, "Rows", None),
        "Columns": getattr(dcm, "Columns", None),
        "NumberOfFrames": getattr(dcm, "NumberOfFrames", 1),
    }

for file_path in file_paths:
    logger.info("=" * 80)
    logger.info(f"FILE: {file_path}")

    try:
        # Fast read (no pixel data)
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

        # Encoding 
        if "TransferSyntaxUID" in dcm.file_meta:
            syntax = dcm.file_meta.TransferSyntaxUID
            logger.info(f"\nEncoding: {syntax.name}")
        else:
            logger.info("\nEncoding: MISSING")

        # Basic Info 
        info = extract_basic_info(dcm)
        logger.info("\n--- BASIC INFO ---")
        for k, v in info.items():
            logger.info(f"{k}: {v}")

        #  Spacing 
        z, y, x = extract_spacing(dcm)

        logger.info("\n--- SPACING (mm) ---")
        logger.info(f"Z (inter-slice): {z if z is not None else 'MISSING'}")
        logger.info(f"Y (pixel spacing): {y if y is not None else 'MISSING'}")
        logger.info(f"X (pixel spacing): {x if x is not None else 'MISSING'}")

        #  Derived 
        if z is not None and y is not None:
            ratio = z / y
            logger.info(f"\nAnisotropy (Z/Y): {ratio:.2f}")

        #  Pixel Data Presence
        logger.info("\n--- PIXEL DATA ---")
        logger.info("Present" if "PixelData" in dcm else "Not loaded (skipped)")

    except Exception as e:
        logger.info(f"ERROR: {e}")