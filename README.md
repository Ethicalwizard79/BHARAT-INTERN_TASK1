# House Price Prediction

This project aims to develop a machine learning model for predicting house prices using Python, scikit-learn, and TensorFlow. The dataset used for this task is the popular Boston House Prices dataset from Kaggle.

## Project Overview

The goal of this project is to build a robust machine learning model that can accurately predict house prices based on various features such as the number of rooms, location, and other relevant factors. The project follows a structured approach, including data acquisition, exploration, preprocessing, model building, training, and evaluation.

## Dataset

The [Boston House Prices dataset](https://www.kaggle.com/c/boston-housing) from Kaggle is used for this project. The dataset contains information about various houses in the Boston area, including features like the number of rooms, crime rate, and distance to employment centers. The target variable is the median value of owner-occupied homes (MEDV).

## Approaches

This project explores two different approaches for building and training the machine learning models:

1. **Scikit-learn**: In this approach, various regression models from the scikit-learn library, such as Linear Regression and Random Forest Regressor, are trained on the preprocessed data. Model performance is evaluated using metrics like Mean Squared Error (MSE) and R-squared. Cross-validation and hyperparameter tuning techniques are also applied to optimize the models.

2. **TensorFlow**: The TensorFlow approach involves building and training deep neural network models. The model architecture, including input layers, hidden layers, and output layers, is defined, and the model is compiled with appropriate loss functions and optimizers. The models are trained on the preprocessed data, and their performance is evaluated on the test set.

## Results

The project compares the performance of the different models trained using scikit-learn and TensorFlow approaches. The best-performing model is selected based on evaluation metrics and further analyzed for future deployments.

## Getting Started

To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/house-price-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open the Jupyter Notebook file: `jupyter notebook house_price_prediction.ipynb`
4. Follow the instructions in the notebook to execute the code and explore the results.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
