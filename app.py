from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier().fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [features]
    prediction = clf.predict(final_features)
    output = iris.target_names[prediction[0]]
    
    return render_template('index.html', prediction_text=f'Predicted Iris Species: {output}')

if __name__ == "__main__":
    app.run(debug=True)