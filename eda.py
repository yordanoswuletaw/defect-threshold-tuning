import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
import json

# Read the datasets
anno_df = pd.read_csv('data/anno_df.csv')
pred_df = pd.read_csv('data/pred_df.csv')

def basic_dataset_info(df, name):
    """Print basic information about the dataset"""
    print(f"\n=== {name} Dataset Analysis ===")
    print("\nBasic Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst few rows:")
    print(df.head())
    
def analyze_confidence_scores(pred_df):
    """Analyze the distribution of confidence scores"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pred_df, x='confidence', bins=50)
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.savefig('confidence_distribution.png')
    plt.close()
    
    print("\nConfidence Score Statistics:")
    print(pred_df['confidence'].describe())

def analyze_class_distribution(df, name):
    """Analyze the distribution of defect classes"""
    if 'class' in df.columns:
        plt.figure(figsize=(10, 6))
        df['class'].value_counts().plot(kind='bar')
        plt.title(f'Class Distribution in {name}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{name.lower()}_class_distribution.png')
        plt.close()
        
        print(f"\nClass Distribution in {name}:")
        print(df['class'].value_counts())

def calculate_polygon_areas(df):
    """Calculate areas of polygons"""
    areas = []
    for idx, row in df.iterrows():
        try:
            # Assuming polygon coordinates are stored in a column as a string representation
            coords = json.loads(row['polygon'])  # Adjust column name if different
            if coords:
                poly = Polygon(coords)
                areas.append(poly.area)
            else:
                areas.append(0)
        except:
            areas.append(0)
    return areas

def main():
    # Basic dataset information
    basic_dataset_info(anno_df, "Annotations")
    basic_dataset_info(pred_df, "Predictions")
    
    # Analyze confidence scores in predictions
    analyze_confidence_scores(pred_df)
    
    # Analyze class distributions
    analyze_class_distribution(anno_df, "Annotations")
    analyze_class_distribution(pred_df, "Predictions")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total annotations: {len(anno_df)}")
    print(f"Total predictions: {len(pred_df)}")
    
    # If we have image IDs, analyze defects per image
    if 'image_id' in anno_df.columns:
        print("\nDefects per Image Statistics:")
        defects_per_image = anno_df['image_id'].value_counts()
        print(defects_per_image.describe())

if __name__ == "__main__":
    main()
