import pandas as pd

def create_image_metrics(anno_df, pred_df):
    """Create image metrics"""
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
    image_metrics_df.to_csv('../data/image_metrics.csv', index=False)
    print("Image metrics created and saved to data/image_metrics.csv")
    return image_metrics_df
