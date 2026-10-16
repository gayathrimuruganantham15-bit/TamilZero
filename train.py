import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CREATE DEMO DATASET
# ==========================================

human_texts = [
    "தமிழ் மிகவும் பழமையான மற்றும் சிறப்புமிக்க மொழியாகும்.",
    "தமிழ்நாட்டில் விவசாயம் மக்களின் முக்கியமான தொழில்களில் ஒன்றாக உள்ளது.",
    "சென்னை தமிழ்நாட்டின் தலைநகரமாகும்.",
    "பள்ளி மாணவர்கள் தினமும் புத்தகங்களை படித்து அறிவை வளர்த்துக் கொள்ள வேண்டும்.",
    "தமிழ் இலக்கியத்தில் பல சிறந்த படைப்புகள் உள்ளன.",
    "மழை பெய்ததால் விவசாய நிலங்களில் நீர் தேங்கியது.",
    "கல்வி என்பது மனித வாழ்க்கையில் மிகவும் முக்கியமானதாகும்.",
    "நமது சுற்றுச்சூழலை பாதுகாப்பது ஒவ்வொருவரின் கடமையாகும்.",
    "தமிழர்களின் பாரம்பரியத்தில் பல சிறப்பான கலைகள் உள்ளன.",
    "கிராமங்களில் மக்கள் இயற்கையுடன் இணைந்து வாழ்கின்றனர்.",
    "நூலகம் மாணவர்களின் அறிவை வளர்க்கும் சிறந்த இடமாகும்.",
    "தமிழ்நாட்டில் பல வரலாற்றுச் சிறப்புமிக்க கோவில்கள் உள்ளன.",
    "காலை நேரத்தில் பூங்காவில் மக்கள் நடைப்பயிற்சி மேற்கொண்டனர்.",
    "விவசாயிகள் கடினமாக உழைத்து உணவு உற்பத்தி செய்கின்றனர்.",
    "மாணவர்கள் ஆசிரியர்களை மதித்து அவர்களின் அறிவுரைகளை பின்பற்ற வேண்டும்.",
    "நமது நாட்டின் வளர்ச்சியில் இளைஞர்களின் பங்கு முக்கியமானது.",
    "மரங்களை வளர்ப்பது சுற்றுச்சூழலுக்கு மிகவும் பயனுள்ளதாகும்.",
    "கடற்கரையில் மாலை நேரத்தில் மக்கள் மகிழ்ச்சியாக இருந்தனர்.",
    "தமிழ் மொழியில் பல அழகான கவிதைகள் எழுதப்பட்டுள்ளன.",
    "உடற்பயிற்சி உடலையும் மனதையும் ஆரோக்கியமாக வைத்திருக்க உதவுகிறது."
]

ai_texts = [
    "தமிழ் மொழி அதன் வளமான வரலாறு மற்றும் தனித்துவமான பண்பாட்டு பாரம்பரியத்தின் காரணமாக உலகளவில் முக்கியத்துவம் பெற்றுள்ளது.",
    "நவீன தொழில்நுட்பத்தின் வளர்ச்சி கல்வித் துறையில் பல்வேறு புதிய வாய்ப்புகளை உருவாக்கி மாணவர்களின் கற்றல் அனுபவத்தை மேம்படுத்துகிறது.",
    "செயற்கை நுண்ணறிவு எதிர்கால சமூகத்தின் பல்வேறு துறைகளில் மாற்றத்தை ஏற்படுத்தக்கூடிய சக்திவாய்ந்த தொழில்நுட்பமாக விளங்குகிறது.",
    "சுற்றுச்சூழல் பாதுகாப்பு என்பது எதிர்கால தலைமுறைகளின் நலனை உறுதி செய்வதற்கான அனைவரின் கூட்டு பொறுப்பாகும்.",
    "கல்வி மனிதர்களின் அறிவு மற்றும் திறன்களை மேம்படுத்துவதோடு சமூகத்தின் ஒட்டுமொத்த வளர்ச்சிக்கும் முக்கிய பங்களிப்பை வழங்குகிறது.",
    "தொழில்நுட்ப முன்னேற்றம் மனித வாழ்க்கையை எளிதாக்குவதுடன் பல்வேறு புதிய வாய்ப்புகளையும் உருவாக்குகிறது.",
    "தமிழ் இலக்கியம் அதன் ஆழமான கருத்துக்கள் மற்றும் அழகிய மொழிநடையின் மூலம் பல தலைமுறைகளாக மக்களை கவர்ந்து வருகிறது.",
    "டிஜிட்டல் உலகின் வளர்ச்சி தகவல்களை வேகமாக பகிர்ந்து கொள்ளவும் உலகின் பல பகுதிகளில் உள்ள மக்களுடன் தொடர்பு கொள்ளவும் உதவுகிறது.",
    "இன்றைய மாணவர்கள் தொழில்நுட்ப அறிவுடன் படைப்பாற்றல் மற்றும் விமர்சன சிந்தனை திறன்களையும் வளர்த்துக் கொள்ள வேண்டும்.",
    "செயற்கை நுண்ணறிவு அடிப்படையிலான கருவிகள் கல்வி மற்றும் ஆராய்ச்சி துறைகளில் புதிய மாற்றங்களை ஏற்படுத்துகின்றன.",
    "நிலையான வளர்ச்சியை உருவாக்க இயற்கை வளங்களை பொறுப்புடன் பயன்படுத்துவது மிகவும் அவசியமாகும்.",
    "சமூக ஊடகங்களின் பயன்பாடு அதிகரித்து வருவதால் தகவல்களின் உண்மைத் தன்மையை சரிபார்ப்பது முக்கியமானதாகியுள்ளது.",
    "தமிழ் போன்ற குறைந்த வளம் கொண்ட மொழிகளுக்கான செயற்கை நுண்ணறிவு கருவிகளின் வளர்ச்சி மொழி தொழில்நுட்பத்தில் புதிய வாய்ப்புகளை உருவாக்கும்.",
    "எதிர்காலத்தில் செயற்கை நுண்ணறிவு மனிதர்களின் அன்றாட செயல்பாடுகளுடன் மேலும் நெருக்கமாக ஒருங்கிணைக்கப்படும்.",
    "தகவல் தொழில்நுட்பத்தின் வளர்ச்சி வேலைவாய்ப்பு மற்றும் தொழில் முனைவோருக்கான புதிய பாதைகளை உருவாக்கியுள்ளது.",
    "நவீன கல்வி முறையில் தனிப்பயனாக்கப்பட்ட கற்றல் மாணவர்களின் திறன்களை சிறப்பாக மேம்படுத்த உதவுகிறது.",
    "சுற்றுச்சூழல் மாற்றங்களை எதிர்கொள்ள தொழில்நுட்பம் மற்றும் சமூக விழிப்புணர்வு இணைந்து செயல்பட வேண்டும்.",
    "மொழி தொழில்நுட்பத்தின் முன்னேற்றம் உலகின் பல்வேறு மொழிகளில் தகவல் அணுகலை எளிதாக்குகிறது.",
    "AI உருவாக்கும் உள்ளடக்கத்தின் வளர்ச்சியால் டிஜிட்டல் தகவல்களின் நம்பகத்தன்மையை மதிப்பிடுவது அவசியமாகியுள்ளது.",
    "புதிய தொழில்நுட்பங்களை பொறுப்புடன் பயன்படுத்துவது சமூகத்தின் நீண்டகால வளர்ச்சிக்கு முக்கியமானதாகும்."
]

# Create dataframe
df_human = pd.DataFrame({
    "text": human_texts,
    "label": "human"
})

df_ai = pd.DataFrame({
    "text": ai_texts,
    "label": "ai"
})

df = pd.concat([df_human, df_ai], ignore_index=True)

# Save dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/tamil_dataset.csv", index=False, encoding="utf-8-sig")

print("=" * 60)
print("TAMILZERO - TRAINING")
print("=" * 60)

print(f"\nDataset size: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 3. LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    pos_label="ai"
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print(f"Accuracy : {logistic_accuracy:.4f}")
print(f"F1 Score : {logistic_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


# ==========================================
# 4. SVM
# ==========================================

svm_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC()
    )
])

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

svm_f1 = f1_score(
    y_test,
    svm_predictions,
    pos_label="ai"
)

print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE")
print("=" * 60)

print(f"Accuracy : {svm_accuracy:.4f}")
print(f"F1 Score : {svm_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_predictions
    )
)


# ==========================================
# 5. SELECT BEST MODEL
# ==========================================

if svm_f1 >= logistic_f1:
    best_model = svm_model
    best_predictions = svm_predictions
    best_name = "SVM"
else:
    best_model = logistic_model
    best_predictions = logistic_predictions
    best_name = "Logistic Regression"

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Selected Model: {best_name}")
print(
    f"Accuracy: "
    f"{accuracy_score(y_test, best_predictions):.4f}"
)


# ==========================================
# SAVE BOTH MODELS
# ==========================================

os.makedirs("models", exist_ok=True)

# Save SVM
joblib.dump(
    svm_model,
    "models/svm_model.pkl"
)

# Save Logistic Regression
joblib.dump(
    logistic_model,
    "models/logistic_model.pkl"
)

# Save selected model
joblib.dump(
    best_model,
    "models/tamilzero_model.pkl"
)

print("\nModels saved:")
print("models/svm_model.pkl")
print("models/logistic_model.pkl")
print("models/tamilzero_model.pkl")


# ==========================================
# MODEL PERFORMANCE
# ==========================================

best_accuracy = accuracy_score(
    y_test,
    best_predictions
)

best_precision = precision_score(
    y_test,
    best_predictions,
    pos_label="ai",
    zero_division=0
)

best_recall = recall_score(
    y_test,
    best_predictions,
    pos_label="ai",
    zero_division=0
)

best_f1 = f1_score(
    y_test,
    best_predictions,
    pos_label="ai",
    zero_division=0
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["human", "ai"]
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# SAVE EVALUATION
# ==========================================

os.makedirs("results", exist_ok=True)

with open(
    "results/evaluation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("TamilZero Model Evaluation\n")
    file.write("===========================\n\n")

    file.write(f"Best Model: {best_name}\n")
    file.write(f"Accuracy: {best_accuracy:.4f}\n")
    file.write(f"Precision: {best_precision:.4f}\n")
    file.write(f"Recall: {best_recall:.4f}\n")
    file.write(f"F1 Score: {best_f1:.4f}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(cm))


print("\nEvaluation saved:")
print("results/evaluation.txt")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["human", "ai"]
)

print("\nConfusion Matrix:")
print(cm)

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)