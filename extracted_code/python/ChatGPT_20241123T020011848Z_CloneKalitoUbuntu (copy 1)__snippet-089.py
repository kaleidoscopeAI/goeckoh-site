object_detection_community = Community("object_detection")

for node in nodes:
    if "object_detection" in node.capabilities:
        object_detection_community.add_member(node)

object_detection_community.share_knowledge("training_data", "Annotated dataset")

