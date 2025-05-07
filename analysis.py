import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.metrics import precision_recall_curve, auc
import ast

# Helper functions
def string_to_coords(xy_str):
    """Convert string coordinates to list of tuples"""
    try:
        # Handle string representations of lists
        if xy_str.startswith('['):
            coords = ast.literal_eval(xy_str)
            return list(zip(coords[::2], coords[1::2]))
        
        # Handle comma-separated string
        coords = [float(x) for x in xy_str.split(',')]
        return list(zip(coords[::2], coords[1::2]))
    except:
        return None 

def calculate_iou(poly1_coords, poly2_coords):
    """Calculate IoU between two polygons"""
    try:
        poly1 = Polygon(poly1_coords)
        poly2 = Polygon(poly2_coords)
        
        if not (poly1.is_valid and poly2.is_valid):
            return 0
        
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        if union_area == 0:
            return 0
            
        return intersection_area / union_area
    except:
        return 0

def evaluate_threshold(anno_df, pred_df, confidence_threshold, iou_threshold=0.5):
    """Evaluate model performance at a given confidence threshold"""
    # Filter predictions by confidence
    filtered_preds = pred_df[pred_df['confidence'] >= confidence_threshold]
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Group by image_id for faster processing
    pred_by_image = filtered_preds.groupby('image_id')
    anno_by_image = anno_df.groupby('image_id')
    
    processed_images = set(anno_df['image_id'].unique())
    
    for image_id in processed_images:
        image_preds = pred_by_image.get_group(image_id) if image_id in pred_by_image.groups else pd.DataFrame()
        image_annos = anno_by_image.get_group(image_id) if image_id in anno_by_image.groups else pd.DataFrame()
        
        matched_annos = set()
        
        # For each prediction in the image
        for _, pred in image_preds.iterrows():
            pred_coords = string_to_coords(pred['xy'])
            if not pred_coords:
                continue
                
            best_iou = 0
            best_anno_idx = None
            
            # Find the best matching annotation
            for idx, anno in image_annos.iterrows():
                if idx in matched_annos:
                    continue
                    
                anno_coords = string_to_coords(anno['xy'])
                if not anno_coords:
                    continue
                
                iou = calculate_iou(pred_coords, anno_coords)
                if iou > best_iou:
                    best_iou = iou
                    best_anno_idx = idx
            
            # If we found a match above the IoU threshold
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_annos.add(best_anno_idx)
            else:
                false_positives += 1
        
        # Count unmatched annotations as false negatives
        false_negatives += len(image_annos) - len(matched_annos)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': confidence_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# Load datasets
anno_df = pd.read_csv('data/anno_df.csv')
pred_df = pd.read_csv('data/pred_df.csv')

# Evaluate range of thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
results = []

for threshold in thresholds:
    print(f"Evaluating threshold: {threshold:.1f}")
    result = evaluate_threshold(anno_df, pred_df, threshold)
    results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot metrics vs threshold
plt.figure(figsize=(10, 6))
plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
plt.plot(results_df['threshold'], results_df['f1'], label='F1 Score')
plt.xlabel('Confidence Threshold')
plt.ylabel('Score')
plt.title('Model Performance Metrics vs Confidence Threshold')
plt.legend()
plt.grid(True)
plt.savefig('performance_metrics.png')
plt.close()

# Find optimal threshold (maximum F1 score)
optimal_result = results_df.loc[results_df['f1'].idxmax()]
print("\nOptimal Threshold Results:")
print(f"Confidence Threshold: {optimal_result['threshold']:.3f}")
print(f"Precision: {optimal_result['precision']:.3f}")
print(f"Recall: {optimal_result['recall']:.3f}")
print(f"F1 Score: {optimal_result['f1']:.3f}")

# Additional analysis for improvements
print("\nAnalysis for Improvements:")

# 1. Class distribution
print("\nClass distribution in annotations:")
print(anno_df['label'].value_counts())

# 2. Confidence distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=pred_df, x='confidence', bins=50)
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.savefig('confidence_distribution.png')
plt.close()

# 3. Image-wise performance
image_metrics = []
for image_id in anno_df['image_id'].unique():
    image_annos = anno_df[anno_df['image_id'] == image_id]
    image_preds = pred_df[pred_df['image_id'] == image_id]
    
    if len(image_preds) > 0:
        avg_conf = image_preds['confidence'].mean()
        n_preds = len(image_preds)
        n_annos = len(image_annos)
        
        image_metrics.append({
            'image_id': image_id,
            'avg_confidence': avg_conf,
            'n_predictions': n_preds,
            'n_annotations': n_annos,
            'pred_anno_ratio': n_preds / n_annos if n_annos > 0 else float('inf')
        })

image_metrics_df = pd.DataFrame(image_metrics)

# Save analysis results
results_df.to_csv('threshold_results.csv', index=False)
image_metrics_df.to_csv('image_metrics.csv', index=False)
