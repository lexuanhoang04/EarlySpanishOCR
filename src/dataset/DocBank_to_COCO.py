import os
import json
import pandas as pd
import traceback
from pathlib import Path
from tqdm import tqdm
import argparse

class COCOData:
    def __init__(self, max_images=None):
        self.src_file_path = []
        self.coco_file_path = []
        self.src_dictionary = []
        self.coco_dictionary = []
        self.max_images = max_images

    def read_src_folder(self, src_path, dest_path):
        for i, path in enumerate(tqdm(Path(src_path).rglob('*.txt'), desc="Loading Source Files")):
            if self.max_images is not None and i >= self.max_images:
                break
            file = pd.read_table(path, header=None, names=[
                "token", "x0", "y0", "x1", "y1", "R", "G", "B", "name", "label"
            ])
            coco_file_path = str(path).replace(".txt", ".json").replace(src_path, dest_path)

            self.src_file_path.append(str(path))
            self.src_dictionary.append(file)
            self.coco_file_path.append(str(coco_file_path))

    def create_dict_layout(self):
        return {
            "info": {
                "year": "",
                "version": "1",
                "description": "",
                "contributor": "",
                "url": "",
                "date_created": "",
            },
            "licenses": [],
            "categories": [
                {"id": 0, "name": "text", "supercategory": ""},
                {"id": 1, "name": "other", "supercategory": ""}
            ],
            "images": [],
            "annotations": []
        }

    def set_image_properties(self, file_name, image_id):
        _, image_name = os.path.split(file_name)
        return {
            "id": image_id,
            "license": "",
            "file_name": image_name.strip(".jpg") + "_ori.jpg",
            "height": "",
            "width": "",
            "date_captured": "",
        }

    def set_object_properties(self, doc_object, doc_object_id, image_id):
        text_labels = {'abstract', 'paragraph', 'title', 'author'}
        original_label = doc_object[9].lower()
        category_id = 0 if original_label in text_labels else 1

        object_width = doc_object[3] - doc_object[1]
        object_height = doc_object[4] - doc_object[2]

        return {
            "id": doc_object_id,
            "image_id": image_id,
            "iscrowd": 0,
            "segmentation": [],
            "category_id": category_id,
            "bbox": [
                int(doc_object[1]),
                int(doc_object[2]),
                int(object_width),
                int(object_height)
            ],
            "area": int(object_width * object_height)
        }

    def convert_to_coco(self):
        try:
            image_id = 0
            doc_object_id = 0
            for i in tqdm(range(len(self.src_file_path)), desc="Converting to COCO"):
                if self.max_images is not None and image_id >= self.max_images:
                    break

                json_dict = self.create_dict_layout()
                image_dict = self.set_image_properties(
                    os.path.split(self.coco_file_path[i])[1].replace(".json", ".jpg"),
                    image_id
                )

                for doc_object in self.src_dictionary[i].values:
                    object_dict = self.set_object_properties(doc_object, doc_object_id, image_id)
                    json_dict["annotations"].append(object_dict)
                    doc_object_id += 1

                json_dict["images"].append(image_dict)
                self.coco_dictionary.append(json_dict)
                image_id += 1
        except:
            traceback.print_exc()

    def save_coco_dataset(self):
        try:
            for i in tqdm(range(len(self.coco_file_path))):
                coco_file_dir = os.path.split(self.coco_file_path[i])[0]
                Path(coco_file_dir).mkdir(parents=True, exist_ok=True)

                with open(self.coco_file_path[i], mode="w") as output_file:
                    output_file.writelines(json.dumps(self.coco_dictionary[i], indent=4))
        except:
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Convert DocBank to COCO format")
    parser.add_argument("--src", type=str, required=True, help="Source folder with DocBank .txt files")
    parser.add_argument("--dest", type=str, required=True, help="Destination folder to save COCO .json files")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to convert")
    args = parser.parse_args()

    coco_converter = COCOData(max_images=args.max_images)
    coco_converter.read_src_folder(args.src, args.dest)
    coco_converter.convert_to_coco()
    coco_converter.save_coco_dataset()


if __name__ == "__main__":
    main()
