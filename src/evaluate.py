from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import argparse
from get_data import get_data
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def m_evaluate(config_file):
    config = get_data(config_file)
    
    # Load model
    model_path = 'saved_models/trained.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)

    # Load test data
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set = config['model']['test_path']

    if not os.path.exists(te_set):
        raise FileNotFoundError(f"Test dataset path not found: {te_set}")

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = test_gen.flow_from_directory(te_set,
                                            target_size=(225, 225),
                                            batch_size=batch,
                                            class_mode=class_mode,
                                            shuffle=False)

    # Get class labels
    label_map = test_set.class_indices
    print(f"Label Mapping: {label_map}")

    # Predictions
    Y_pred = model.predict(test_set, steps=len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)

    # Create reports folder if it doesn't exist
    os.makedirs('reports', exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(test_set.classes, y_pred), annot=True, fmt='d', cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('reports/Confusion_Matrix.png')  # Save as PNG for easy viewing
    plt.close()

    # Classification Report
    target_names = list(label_map.keys())
    df = pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True)).T
    df['support'] = df['support'].astype(int)
    df.to_csv('reports/classification_report.csv')  # Save as CSV for readability

    print('Classification Report and Confusion Matrix are saved in the "reports" folder.')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    passed_args = args_parser.parse_args()
    m_evaluate(config_file=passed_args.config)
