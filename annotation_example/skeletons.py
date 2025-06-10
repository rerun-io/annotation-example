from rtmlib.visualization import coco133, coco17

COCO_133_IDS: list[int] = [id for id, kpt_info in coco133["keypoint_info"].items()]
link_dict = {}
for _, kpt_info in coco133["keypoint_info"].items():
    link_dict[kpt_info["name"]] = kpt_info["id"]

COCO_133_LINKS: list[tuple[int, int]] = []
for _, ske_info in coco133["skeleton_info"].items():
    link = ske_info["link"]
    COCO_133_LINKS.append((link_dict[link[0]], link_dict[link[1]]))

COCO_133_ID2NAME: dict[int, str] = {id: kpt_info["name"] for id, kpt_info in coco133["keypoint_info"].items()}


COCO_17_IDS: list[int] = [id for id, kpt_info in coco17["keypoint_info"].items()]
link_dict_17 = {}
for _, kpt_info in coco17["keypoint_info"].items():
    link_dict_17[kpt_info["name"]] = kpt_info["id"]

COCO_17_LINKS: list[tuple[int, int]] = []
for _, ske_info in coco17["skeleton_info"].items():
    link = ske_info["link"]
    COCO_17_LINKS.append((link_dict_17[link[0]], link_dict_17[link[1]]))

COCO_17_ID2NAME: dict[int, str] = {id: kpt_info["name"] for id, kpt_info in coco17["keypoint_info"].items()}
COCO_17_NAME2ID: dict[str, int] = {kpt_info["name"]: id for id, kpt_info in coco17["keypoint_info"].items()}
print("COCO_17_NAME2ID", COCO_17_NAME2ID)
