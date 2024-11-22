# Mobile Price Estimation

This project predicts the price of mobile phones based on their specifications using a Random Forest Regression model. The project also includes statistical visualizations for data analysis and explores relationships between features like RAM, storage, battery capacity, and screen size.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Visualization](#visualization)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Introduction
Accurate pricing of mobile phones is essential for both manufacturers and customers. This project leverages machine learning to predict mobile phone prices using key technical specifications.

---

## Features
- Predict mobile phone prices based on specifications such as:
  - **RAM**
  - **Storage**
  - **Screen Size**
  - **Battery Capacity**
- Perform exploratory data analysis (EDA) with matplotlib visualizations.
- Evaluate model performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.

---

## Dataset
The dataset includes mobile phone specifications and their respective prices. Key columns:
- **Brand**: Manufacturer of the mobile phone.
- **Model**: Model name of the mobile phone.
- **RAM**: Amount of memory (in GB).
- **Storage**: Internal storage capacity (in GB).
- **Screen Size (inches)**: Diagonal screen size (in inches).
- **Battery Capacity (mAh)**: Battery capacity in milliampere-hours.
- **Price ($)**: Price of the mobile phone in USD.

---

## Requirements
Install the required dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas`
- `matplotlib`
- `scikit-learn`

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mobile-price-estimation.git
   cd mobile-price-estimation
   ```

2. Place the dataset file `Mobile phone price.csv` in the same directory as the script.

3. Run the Python script:
   ```bash
   python main.py
   ```

4. Modify the `example` dictionary in the script to predict the price for different specifications:
   ```python
   example = {'RAM': 8, 'Storage': 128, 'Screen Size (inches)': 6.5, 'Battery Capacity (mAh)': 4500}
   ```

---

## Model Performance
- **Mean Absolute Error (MAE):** 83.74
- **Mean Squared Error (MSE):** 15562.00
- **R² Score:** 0.825

---

## Visualization
The script generates the following plots:
1. **Price Distribution**: Histogram showing the spread of mobile prices.
2. **Price vs RAM**: Scatter plot illustrating the relationship between RAM and price.
3. **Price vs Battery Capacity**: Scatter plot showing the impact of battery capacity on price.

## Future Improvements
- Add support for categorical features like `Brand`.
- Tune hyperparameters for improved accuracy.
- Explore alternative models like Gradient Boosting or Neural Networks.
- Deploy the model as a web application using Flask or Streamlit.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
