import json
from tqdm import tqdm


def convert_segmentation_to_polygon(segmentation):
    return [
        [segmentation[0][i], segmentation[0][i+1]]
        for i in range(0, len(segmentation[0]), 2)
    ]


def convert_segmentations_to_polygons(segmentations):
    return [
        [
            [segmentation[0][i], segmentation[0][i+1]]
            for i in range(0, len(segmentation[0]), 2)
        ]
        for segmentation in segmentations
    ]


def convert_annotations_to_metadata(annotations, pbar):
    metadata = {}

    current_img_metadata = []
    current_img_id = None

    for annotation in annotations:
        pbar.update()

        annotation_metadata = {}
        annotation_metadata["polygon"] = convert_segmentation_to_polygon(annotation["segmentation"])
        annotation_metadata["box"] = annotation["bbox"]
        annotation_metadata["box"] = [
            min(max(int(annotation_metadata["box"][0]), 0), 1023),
            min(max(int(annotation_metadata["box"][1]), 0), 1023),
            int(annotation_metadata["box"][2]),
            int(annotation_metadata["box"][3])
        ]

        if str(annotation["image_id"]) == current_img_id:
            current_img_metadata.append(annotation_metadata)
            continue

        if current_img_id is not None:
            metadata[current_img_id] = current_img_metadata
            
        current_img_id = str(annotation["image_id"])
        current_img_metadata = []
        current_img_metadata.append(annotation_metadata)

    metadata[current_img_id] = current_img_metadata

    return metadata


def preprocess_annotations(dataset_path: str="./data/WHU-Building-Dataset"):
    metadata = {}

    with open(f"{dataset_path}/annotation/train.json") as f:
        train = json.load(f)

    with open(f"{dataset_path}/annotation/test.json") as f:
        test = json.load(f)

    with open(f"{dataset_path}/annotation/validation.json") as f:
        validation = json.load(f)

    pbar = tqdm(total = len(train["annotations"]) + len(test["annotations"]) + len(validation["annotations"]))

    metadata.update(convert_annotations_to_metadata(train["annotations"], pbar))
    metadata.update(convert_annotations_to_metadata(test["annotations"], pbar))
    metadata.update(convert_annotations_to_metadata(validation["annotations"], pbar))

    with open(f"{dataset_path}/metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    preprocess_annotations()
