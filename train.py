from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = fetch_olivetti_faces()

X = data.data
y = data.target

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "savedmodel.pth")

print("Model trained and saved successfully!")