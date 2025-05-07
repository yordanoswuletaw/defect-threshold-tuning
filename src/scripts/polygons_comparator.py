import pandas as pd
from tqdm import tqdm
from coordinates_parser import parse_coordinates
from iou_calculater import calculate_iou

def match_polygons(anno_df, pred_df, iou_threshold=0.5):
    """Match predictions to annotations based on IoU"""
    results = []
    
    # Process each image separately
    for image_id in tqdm(anno_df['image_id'].unique()):        
        # Get annotations and predictions for this image
        image_annos = anno_df[anno_df['image_id'] == image_id]
        image_preds = pred_df[pred_df['image_id'] == image_id]
        
        # Convert coordinates to polygon format
        anno_polygons = []
        for _, anno in image_annos.iterrows():
            coords = parse_coordinates(anno['xy'])
            if coords:
                anno_polygons.append({
                    'coords': coords,
                    'id': anno['id'],
                    'class': anno['defect_class_id'],
                    'matched': False
                })
        
        pred_polygons = []
        for _, pred in image_preds.iterrows():
            coords = parse_coordinates(pred['xy'])
            if coords:
                pred_polygons.append({
                    'coords': coords,
                    'id': pred['prediction_id'],
                    'confidence': pred['confidence'],
                    'class': pred['prediction_class'],
                    'matched': False
                })
        
        # Calculate IoU between all pairs
        for pred in pred_polygons:
            best_iou = 0
            best_anno = None
            
            for anno in anno_polygons:
                if not anno['matched']:  # Only consider unmatched annotations
                    iou = calculate_iou(pred['coords'], anno['coords'])
                    if iou > best_iou:
                        best_iou = iou
                        best_anno = anno
            
            # If we found a match above threshold
            if best_iou >= iou_threshold and best_anno is not None:
                best_anno['matched'] = True
                results.append({
                    'image_id': image_id,
                    'anno_id': best_anno['id'],
                    'pred_id': pred['id'],
                    'anno_class': best_anno['class'],
                    'pred_class': pred['class'],
                    'confidence': pred['confidence'],
                    'iou': best_iou,
                    'match_type': 'true_positive'
                })
            else:
                # False positive
                results.append({
                    'image_id': image_id,
                    'anno_id': None,
                    'pred_id': pred['id'],
                    'anno_class': None,
                    'pred_class': pred['class'],
                    'confidence': pred['confidence'],
                    'iou': best_iou,
                    'match_type': 'false_positive'
                })
        
        # Add unmatched annotations as false negatives
        for anno in anno_polygons:
            if not anno['matched']:
                results.append({
                    'image_id': image_id,
                    'anno_id': anno['id'],
                    'pred_id': None,
                    'anno_class': anno['class'],
                    'pred_class': None,
                    'confidence': None,
                    'iou': 0,
                    'match_type': 'false_negative'
                })
    
    matches_df = pd.DataFrame(results)
    # Save results
    matches_df.to_csv('../data/polygon_matches.csv', index=False)
    print("Polygon matches created and saved to data/polygon_matches.csv")
    return matches_df