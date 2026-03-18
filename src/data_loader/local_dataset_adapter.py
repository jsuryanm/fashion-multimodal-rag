from pathlib import Path 
import pandas as pd 
from PIL import Image 
import logging 

logger = logging.getLogger(__name__)

CATALOGUE = [
    {
        "filename": "img1.png",
        "items": [
            {"Item Name": "Black Velvet Evening Gown",        "Price": "249.99", "Link": "https://example.com/p/1001"},
            {"Item Name": "Structured Longline Blazer Coat",  "Price": "189.99", "Link": "https://example.com/p/1002"},
            {"Item Name": "Crystal Drop Evening Earrings",    "Price": "69.99",  "Link": "https://example.com/p/1003"},
        ],
    },
    {
        "filename": "img2.png",
        "items": [
            {"Item Name": "Pleated Cape Sleeve Midi Dress",   "Price": "279.99", "Link": "https://example.com/p/2001"},
            {"Item Name": "Tailored Grey Slim Fit Suit",      "Price": "399.99", "Link": "https://example.com/p/2002"},
            {"Item Name": "Grey Turtleneck Dress Shirt",      "Price": "89.99",  "Link": "https://example.com/p/2003"},
        ],
    },
    {
        "filename": "img3.png",
        "items": [
            {"Item Name": "Black Double Breasted Overcoat",   "Price": "229.99", "Link": "https://example.com/p/3001"},
            {"Item Name": "Black Formal Button Shirt",        "Price": "79.99",  "Link": "https://example.com/p/3002"},
            {"Item Name": "Black Tailored Dress Trousers",    "Price": "119.99", "Link": "https://example.com/p/3003"},
        ],
    },
    {
        "filename": "img4.png",
        "items": [
            {"Item Name": "Black Wool Overcoat",              "Price": "199.99", "Link": "https://example.com/p/4001"},
            {"Item Name": "Black Knit Sweater",               "Price": "89.99",  "Link": "https://example.com/p/4002"},
            {"Item Name": "Grey Straight Leg Formal Trousers","Price": "109.99", "Link": "https://example.com/p/4003"},
            {"Item Name": "Leather Structured Tote Bag",      "Price": "159.99", "Link": "https://example.com/p/4004"},
            {"Item Name": "Black Leather Penny Loafers",      "Price": "139.99", "Link": "https://example.com/p/4005"},
        ],
    },
    {
        "filename": "img5.png",
        "items": [
            {"Item Name": "Black Longline Tailored Coat",     "Price": "219.99", "Link": "https://example.com/p/5001"},
            {"Item Name": "Black Buttoned Waistcoat",         "Price": "129.99", "Link": "https://example.com/p/5002"},
            {"Item Name": "Black Slim Fit Dress Pants",       "Price": "119.99", "Link": "https://example.com/p/5003"},
        ],
    },
]

def load_local_dataset(images_dir: str = r"data/"):
    images_path = Path(images_dir)
    rows = []

    for entry in CATALOGUE:
        image_path = images_path / entry["filename"]

        if not image_path.exists():
            logger.warning("Skipping missing image: %s", image_path)

            continue

        try:
            image = Image.open(image_path)
        except Exception as e:
            logger.warning("Could not open: %s", image_path)
            continue
        
        image_url = str(image_path.resolve())
 
        for item in entry["items"]:
            rows.append(
                {
                    "Image URL": image_url,
                    "Item Name": item["Item Name"],
                    "Price": item["Price"],
                    "Link": item["Link"],
                    "image": image,
                }
            )
 
    df = pd.DataFrame(rows)
 
    if df.empty:
        logger.error(
            "No images loaded from '%s'. "
            "Check filenames match CATALOGUE entries.",
            images_dir,
        )
    else:
        logger.info(
            "Loaded %d items across %d images from '%s'",
            len(df),
            df["Image URL"].nunique(),
            images_dir,
        )
 
    return df

# if __name__ == "__main__":
#     df = load_local_dataset()
#     print(df.head())