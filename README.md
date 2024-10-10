
Project Name: Data Science Work

Repository Overview

This repository contains the implementation of various files related to data science workflows, focusing on tools like Metaflow, Great Expectations, model monitoring, and deployment. Below is a breakdown of the key components:

Folder Structure

- app/  
  Contains FastAPI or Streamlit apps for serving models and predictions.

- data/  
  Directory containing datasets used. 

- gx/  
  Great Expectations configurations and checkpoints for data validation.
  
- metaflow-tools/  
  Tools and utilities for working with Metaflow, including scripts for managing flows and tasks.
  
- metaflow_scripts/  
  Scripts specifically used for Metaflow workflows and orchestration. Updated for Lab 9.

- mltest/  
  Contains tests and experiments with machine learning models. Includes work from labs 7 and 8.

- model_token/  
  Lab 9 scripts for generating model tokens and handling secure credentials.

- notebooks/  
  Jupyter notebooks for experimenting with models and documenting lab exercises. Updated for Lab 9.

- scripts/  
  General-purpose scripts for various tasks like preprocessing, training, and model testing.

Files

- .DS_Store  
  MacOS-generated file for folder configuration. Can be ignored.

- .gitignore  
  Git ignore file, configured to exclude unnecessary or sensitive files from the repository.

- README.md  
  This file provides an overview of the repository.

- dockerfile  
  Docker configuration used to containerize the environment for labs 7 and 8.

- pipeline-activate.sh  
  Shell script for activating the data pipeline used in labs 7 and 8.

- requirements.txt  
  List of Python dependencies required for running the project. Includes packages used in labs 7 and 8.

- server.sh  
  Shell script for running server-related tasks, likely related to model deployment in FastAPI/Streamlit.

---

Lab Details

- Lab 7, 8:
  - Focused on setting up Metaflow workflows and utilizing Great Expectations for data validation.
  - Docker containerization to ensure a consistent environment.
  - Worked on initial model tests and created pipelines for automating deployment.

- Lab 9:
  - Extended previous labs by refining model predictions and deployment processes.
  - Updated FastAPI/Streamlit applications for real-time model serving.
  - Enhanced Metaflow scripts and added new tools for handling model tokens and security.

How to Run the Project

1. Clone the repository:
   git clone https://github.com/your-username/repo-name.git
   cd repo-name

2. Set up the environment:
   Make sure you have Docker installed, then build the image:
   docker build -t lab-env .

3. Activate the pipeline:
   Run the following command to activate the pipeline:
   ./pipeline-activate.sh

4. Run the model server:
   Start the model serving application:
   ./server.sh

5. Run Metaflow scripts:
   Use the Metaflow tools located in metaflow-tools/ to orchestrate workflows:
   python metaflow_scripts/lab9_flow.py

Requirements

- Python 3.8 or higher
- Docker
- Metaflow
- Great Expectations
- FastAPI / Streamlit
- Refer to requirements.txt for additional packages

---

Contributors

- Lance Santerre

For more information or questions, please feel free to reach out!
