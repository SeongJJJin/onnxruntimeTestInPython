import json

def results_to_json(predictions, class_names, conf_threshold=0.25):
    results = []
    for pred in predictions:
        class_index = int(pred[0])
        # class_name = class_names[class_index]
        confidence = float(pred[4])
        if confidence < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, pred[1:5])
        result = {
            # "class_name": class_name,
            "confidence": confidence,
            "box_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        results.append(result)
    return json.dumps(results)