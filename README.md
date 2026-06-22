# Drug–Drug Interaction & Side Effect Predictor using Graph Neural Networks (GNN)

## Overview

This project is a Streamlit-based web application that predicts potential drug-drug interactions and possible side effects using Graph Neural Networks (GNN). The system analyzes medication combinations prescribed to patients and identifies interaction risks, severity levels, and associated side effects.

## Features

* Upload drug interaction datasets in CSV format
* Upload patient medication datasets
* Predict drug-drug interactions using Graph Convolutional Networks (GCN)
* Display interaction probability scores
* Identify severity levels of interactions
* Suggest possible side effects
* Interactive web interface using Streamlit
* Real-time patient medication analysis

## Technologies Used

* Python
* Streamlit
* PyTorch
* PyTorch Geometric
* Pandas
* Scikit-learn
* Graph Neural Networks (GCN)

## Project Structure

```text
Drug-Interaction-GNN/
│
├── app.py
├── drug_interactions.csv
├── patients_dataset.csv
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nagadeepika206/Drug-Interaction-GNN.git
```

2. Navigate to the project directory:

```bash
cd Drug-Interaction-GNN
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Dataset Format

### Drug Interaction Dataset

Required columns:

* drug1
* drug2
* severity
* description

### Patient Dataset

Required columns:

* Name
* Drugs

Example:

```text
Name,Drugs
John,Aspirin;Ibuprofen
Sarah,Paracetamol;Warfarin
```

## Model Architecture

The project uses a Graph Convolutional Network (GCN) consisting of:

* Two GCNConv layers
* ReLU activation function
* Fully connected output layer
* Binary interaction prediction

## Future Enhancements

* Advanced side-effect prediction
* Multi-class interaction severity classification
* Drug recommendation system
* Integration with medical databases
* Explainable AI visualizations
* Cloud deployment

## Author

Nagadeepika

## License

This project is developed for educational and research purposes.
