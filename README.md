# AI-driven-Anomaly-Detection
## Project Description

This project focuses on developing an AI-driven anomaly detection framework for electronic component burn-in and screening processes using machine learning techniques. The primary objective is to identify abnormal behaviour in electronic components during burn-in testing while exploring the possibility of detecting potential failures at an early stage. Traditional burn-in testing involves operating electronic components under controlled stress conditions for a specified duration to identify components that may fail or exhibit unreliable performance. However, conventional testing approaches can involve significant testing time and operational costs.

To address these challenges, this project explores the application of machine learning-based anomaly detection techniques to improve the efficiency of component screening. Two unsupervised machine learning algorithms, Isolation Forest and One-Class Support Vector Machine (One-Class SVM), are implemented using Python to identify deviations from normal component behaviour. These models analyse the available burn-in data and classify observations based on their deviation from expected patterns.

## Key Features

Machine learning-based anomaly detection for electronic component burn-in and screening.

Implementation of Isolation Forest and One-Class SVM algorithms.

Integration of Part Average Testing (PAT) as a conventional comparison method.

Analysis of early anomaly detection versus full-duration burn-in testing.

Evaluation of detection performance and potential testing cost benefits.

Python-based simulation, data processing and result visualisation.

## Technologies Used

Programming Language: Python

Machine Learning: Isolation Forest, One-Class SVM

Conventional Testing Method: Part Average Testing (PAT)

Data Processing: Python-based data handling and feature extraction

Evaluation: Comparative analysis, CSV-based results and graphical visualisation

## Project Status

The project implementation and comparative evaluation have been completed. Preparation of a research paper based on the project methodology and findings is planned.

## How to Run the Project

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/praha2609-alt/RTL-Design-and-Verification.git
cd RTL-Design-and-Verification
```

### 2. Install Python Dependencies

Make sure Python 3.x is installed on your system. Install the required libraries using:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Execute the main application file:

```bash
python app.py
```

### 4. View the Results

Once the program executes, the user can view the generated anomaly detection results and performance comparisons.

The project includes:
- Anomaly detection outputs from Isolation Forest and One-Class SVM.
- Comparison results using Part Average Testing (PAT).
- CSV files containing evaluation results.
- Graphical visualisations of early detection and testing cost comparisons.

**Note:** Ensure that the required dataset is available in the expected project directory before running the application. Run the scripts from the project root directory so that relative file paths resolve correctly.
