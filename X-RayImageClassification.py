#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve

traindata = pd.read_csv(r'C:\Users\rihab\Downloads\DataSets\XRayData\train_df.csv')
testdata = pd.read_csv(r'C:\Users\rihab\Downloads\DataSets\XRayData\test_df.csv')

# %%
# DATA EXPLORATION
# Train Data
print('Columns:\n', traindata.columns)
print('First 5 rows:\n', traindata.head())
print('Description:\n', traindata.describe())
print('Shape:\n', traindata.shape)
print('Count of null values in each column:\n', traindata.isnull().sum())

# %%
# Test Data
print('Columns:\n', testdata.columns)
print('First 5 rows:\n', testdata.head())
print('Description:\n', testdata.describe())
print('Shape:\n', testdata.shape)
print('Count of null values in each column:\n', testdata.isnull().sum())

# %%
# Image Data
from PIL import Image
import matplotlib.pyplot as plt

# Visualize Sample Image
pngimage = Image.open("image_png.png")
pixels = list(pngimage.getdata())
print('Sample Image Pixels:', pixels)
print('Image Size:', pngimage.size)
plt.imshow(pngimage, cmap='gray')


# %%
# Target/Label Exploration

# Count frequency of each label to see which body parts are over or underrepresented
# Flatten the list of labels to count occurrences of each individual label
traindata['Target'] = traindata['Target'].apply(lambda x: list(map(int, x.split())) if isinstance(x, str) else [int(x)])
# Check the first few rows to ensure it looks correct
print(traindata['Target'].head())
# Flatten into one list of all targets for all images
all_labels = [label for sublist in traindata['Target'] for label in sublist]
label_counts = pd.Series(all_labels).value_counts()
print(label_counts)
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.xlabel('Body Part Label')
plt.ylabel('Frequency')
plt.title('Distribution of Body Part Labels')
plt.xticks(rotation=90)
plt.show()
print(traindata['Target'].head(25))

# %%
# Number of Targets/Labels per Image
num_labels_per_image = traindata['Target'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(num_labels_per_image, bins=range(1, num_labels_per_image.max() + 2), kde=False)
plt.xlabel('Number of Labels per Image')
plt.ylabel('Number of Images')
plt.title('Distribution of Number of Labels per Image')
plt.show()
print(f"Max number of labels per image: {num_labels_per_image.max()}")
print(f"Average number of labels per image: {num_labels_per_image.mean()}")

# %%
# Binary matrix showing which body parts often appear in the same image
binary_matrix = np.zeros((len(traindata), len(label_counts)), dtype=int)

for idx, labels in enumerate(traindata['Target']):
    for label in labels:
        binary_matrix[idx, label] = 1

# Compute the co-occurrence matrix (correlation between body part labels)
co_occurrence_matrix = np.dot(binary_matrix.T, binary_matrix)
co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=range(len(label_counts)), columns=range(len(label_counts)))

label_names = {
    0: 'Abdomen', 1: 'Ankle', 2: 'Cervical Spine', 3: 'Chest', 4: 'Clavicles', 
    5: 'Elbow', 6: 'Feet', 7: 'Finger', 8: 'Forearm', 9: 'Hand', 
    10: 'Hip', 11: 'Knee', 12: 'Lower Leg', 13: 'Lumbar Spine', 14: 'Others', 
    15: 'Pelvis', 16: 'Shoulder', 17: 'Sinus', 18: 'Skull', 19: 'Thigh', 
    20: 'Thoracic Spine', 21: 'Wrist'
}
co_occurrence_df_renamed = co_occurrence_df.rename(index=label_names, columns=label_names)

# Plot heatmap of the co-occurrence matrix
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_df_renamed, cmap='YlGnBu', annot=True, fmt='d', square=True)
plt.title('Co-occurrence Matrix of Body Parts')
plt.xlabel('Body Part Label')
plt.ylabel('Body Part Label')
plt.show()

# %%
# DATA PREPROCESSING
# Resize and Augment Images
from pathlib import Path
from fastai.vision.all import *

path = Path(r'C:\Users\rihab\Downloads\DataSets\XRayData')
df = pd.read_csv(path / 'train_df.csv')

df['Target'] = df['Target'].apply(lambda x: list(map(int, x.split())) if isinstance(x, str) else [int(x)])

def get_labels(row):
    return row['Target']

def get_train_image_path(uid):
    return path / f'images/train' / f"{uid}-c.png"

xray_block = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),  
    get_x=lambda row: get_train_image_path(row['SOPInstanceUID']),
    get_y=get_labels,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(224),  # Resize to standard size
    batch_tfms=aug_transforms(size=224, 
                              min_scale=0.75,
                              do_flip=True,  # Enable horizontal flipping
                              flip_vert=False,  # Vertical flips are usually not valid for X-rays
                              max_rotate=5,  # Random rotation between -10 and 10 degrees
                              min_zoom=1.0, max_zoom=.8,  # Zoom in/out within this range
                              max_lighting=0.2,  # Adjust brightness/contrast
                              max_warp=0.1  # Small perspective warping)  # Augmentations
))

dls = xray_block.dataloaders(df)
dls.show_batch(max_n=9, figsize=(8, 8))
dls.one_batch()
print(type(traindata['Target'][0]))
print(type(traindata['Target'][0][0]))
print(all(x in [0, 1] for x in traindata['Target'][0]))

for i, (x, y) in enumerate(dls.train):
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    if i > 5: break

# %%
# MODEL DESIGN AND TRAINING
# Load a pretrained ResNet34 model
learn = vision_learner(dls, resnet50, metrics=[partial(accuracy_multi, thresh=0.5), F1ScoreMulti(thresh=0.5, average='macro'), F1ScoreMulti(thresh=0.5, average='samples'), F1ScoreMulti(thresh=0.5, average='weighted')])
learn.lr_find() 

# %%
# HYPERPARAMETER TUNING
learn.fine_tune(25, base_lr=0.0015)

# %%
# EXPLORE TRAINING RESULTS
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10, 10))
learn.show_results()

# %%
# FINAL TESTING ON TEST SET
test_df = pd.read_csv(path / 'test_df.csv')

def get_test_image_path(uid):
    return path / 'images/test' / f"{uid}-c.png"

test_dl = dls.test_dl([get_test_image_path(uid) for uid in test_df['SOPInstanceUID'].tolist()])
preds = learn.get_preds(dl=test_dl)

# %%
# EXPLORE TEST RESULTS
def get_preds_from_preds(pred_tensor):
    has_tsh = any(pred_tensor >= 0.5)
    if has_tsh: 
        return [learn.dls.vocab[i] for i in range(len(pred_tensor)) if pred_tensor[i] >= 0.5]
    else:
        return [learn.dls.vocab[int(pred_tensor.argmax())]]

def get_labels_from_preds(items):
    if isinstance(items, list):  # Items should be a list of ints
        return " ".join(map(str, sorted(items)))
    elif isinstance(items, (int, float)):
        return str(int(items))
    else:
        return "-1"

labelled_preds = [get_preds_from_preds(pred) for pred in preds[0]]
str_preds = [get_labels_from_preds(items) for items in labelled_preds]

test_df['Target'] = str_preds
df_submission = test_df[['SOPInstanceUID', 'Target']]
print(df_submission.shape)
df_submission.head(20)

# Plot distribution of predicted labels
all_pred_labels = []
for pred in test_df['Target']:
    if pred != "-1":  # Ignore images with no predicted labels
        all_pred_labels.extend(map(int, pred.split()))

all_pred_labels = np.array(all_pred_labels)
label_counts = np.bincount(all_pred_labels, minlength=22)
label_df = pd.DataFrame({'Label': range(22), 'Count': label_counts})
label_df['Label Name'] = label_df['Label'].map(label_names)

plt.figure(figsize=(12, 6))
sns.barplot(x='Label Name', y='Count', data=label_df, palette='viridis')
plt.xlabel('Predicted Body Part Label')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Labels in Test Set')
plt.xticks(rotation=90)
plt.show()

# %%
# SAVE PREDICTIONS
test_df[['SOPInstanceUID', 'Target']].to_csv('test_predictions.csv', index=False)
df_check = pd.read_csv('test_predictions.csv')
print(df_check.head(10))

# %%
