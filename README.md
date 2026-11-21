# TalentSol - Applicant Tracking System Learning Project (Machine Learning Component)

This repository contains the machine learning component of the TalentSol project, focusing on training a supervised classifier to predict and recommend candidates for prioritization for a given job role and description. This is a learning project exploring different text embedding techniques and model evaluation for a real-world recruitment use case.

> **Important: the saved model uses Ethnicity.** The saved logistic regression pipeline one hot encodes an Ethnicity column as an input. The Ethnicity values in the training data are mostly national origin labels, such as Chinese, Irish and Nigerian. Using this input to screen candidates is not acceptable for real hiring. I plan to retrain the model without Ethnicity, Race1 and Race2. Race1 and Race2 are already excluded from the saved model. The reported metrics are scores on the training data. They must be measured again after retraining, with a held out split. This model is a learning artifact. Do not use it for real hiring decisions. See [Fairness](#fairness) and [Limitations](#limitations).

## Project Goal

The primary goal of this project is to train a supervised logistic regression model that can predict which job applicants are the "Best Match" for a given job role and description, thereby helping recruiters prioritize candidates during the screening process.

## Dataset

The project uses the `job_applicant_dataset.csv` dataset, which contains information about job applicants, their resumes, and whether they were considered a "Best Match" for a specific job role and description.

**Dataset Source:** The dataset is sourced from Kaggle: [Recruitment Dataset](https://www.kaggle.com/datasets/surendra365/recruitement-dataset)

The resume text in this dataset follows a fixed template, for example "Proficient in ..., with senior-level experience in the field." I have not confirmed whether the dataset is synthetic. I have not confirmed which license applies to it.

## Preprocessing and Feature Engineering

Before we can train a machine learning model on text data like job descriptions and resumes, we need to convert the words and phrases into a numerical format that the model can understand. This process is called **Feature Engineering**, where we create meaningful numerical features from the raw data.

The following preprocessing and feature engineering steps were performed on the dataset:

1.  **Data Loading:** The `job_applicant_dataset.csv` file was loaded into a pandas DataFrame.
2.  **Identifier and Sensitive Feature Removal:** I removed 'Job Applicant Name', 'Job Applicant ID', 'Age', and 'Gender' early in the preprocessing pipeline. These columns are not direct model inputs. This step does not make the data private. The resume text stays in the model input and in the committed training CSV, and resume text can contain personal information. They were also not considered relevant or appropriate for the prioritization task in this version of the project, as the focus is on text matching between the resume/CV and the job description/job role.
3.  **Handling Missing Values:** Checked for and confirmed no missing data points were present after initial cleaning.
4.  **Oversampling:** The classes were close to balanced, with 5,150 rows that were not a "Best Match" and 4,850 rows that were. I oversampled the "Best Match" class to 5,150 rows by sampling with replacement. This bootstrap sample duplicated some positive rows and left out others. The committed training CSV has 10,300 rows, of which 8,260 are unique. It contains 3,110 unique positive rows, so 1,740 of the 4,850 original positive rows are absent.
5.  **Categorical Encoding:** Job Roles and Ethnicity were one hot encoded. The Race column was split into Race1 and Race2, and then both columns were dropped. The saved model still uses Ethnicity as an input. I originally kept Ethnicity for diversity hiring reasons. That was the wrong choice. A screening model must not score candidates on a protected characteristic. I plan to retrain the model without Ethnicity.
6.  **Text Feature Embedding:** This is where we convert the text from 'Job Description' and 'Resume' into numbers. We explored different methods:
    *   **TF-IDF Vectorization (Term Frequency-Inverse Document Frequency):** Calculates a score for each word based on its frequency within a document and its rarity across all documents. We also explored and optimized **n-grams** (sequences of words) to capture phrase meaning. Bigrams (1,2) raised the best cross validation ROC AUC from 0.5801 to 0.5967. These cross validation scores share the duplicate row problem described below.
    *   **Word2Vec Embeddings:** Learns dense numerical vector representations for words, capturing semantic relationships. Document vectors were created by averaging word vectors.
    *   **Hybrid Embeddings:** Combinations of TF-IDF and Word2Vec features were created to leverage the strengths of both approaches. An optimized hybrid approach used TF-IDF with tuned n-grams and Word2Vec.

## Modeling and Evaluation

A **Logistic Regression** model was chosen as the classifier. Various model configurations (using different text embedding techniques and parameters) were trained and evaluated using **5-fold Stratified Cross-Validation**. This cross-validation approach ensures that each fold has a representative proportion of both classes ('Best Match' and 'Not Best Match'). I ran cross validation on the oversampled data, which contains duplicate rows. Copies of one row can fall in both a training fold and a validation fold, so the cross validation scores are likely optimistic.

Hyperparameter tuning for the Logistic Regression model (specifically the regularization strength `C` and the type of penalty `l1` or `l2`) and, where applicable, TF-IDF vectorizer parameters (like `ngram_range`), was performed using **Grid Search Cross-Validation (`GridSearchCV`)**.

We explored and tuned the following six model configurations:

1.  **Logistic Regression with Tuned TF-IDF (Default N-grams):** Tuned `C` and `penalty` for Logistic Regression using TF-IDF features with a default `ngram_range` of (1,1).
2.  **Logistic Regression with Tuned Word2Vec:** Tuned `C` and `penalty` for Logistic Regression using Word2Vec features.
3.  **Logistic Regression with Tuned, N-gram Optimized TF-IDF:** Tuned the `ngram_range` for the TF-IDF vectorizers (exploring (1,1), (1,2), (1,3)) while using the best `C` and `penalty` found in the initial Tuned TF-IDF model tuning.
4.  **Logistic Regression with Hybrid Embeddings (Default TF-IDF + Word2Vec):** Combined default TF-IDF (ngram\_range=(1,1)) and Word2Vec features, then tuned `C` and `penalty` for the Logistic Regression model on this combined feature set.
5.  **Logistic Regression with Optimized Hybrid Embeddings (Optimized TF-IDF + Word2Vec):** Combined optimized TF-IDF (using the tuned `ngram_range` from configuration 3) and Word2Vec features, then tuned `C` and `penalty` for the Logistic Regression model on this combined feature set.
6.  **Weighted Hybrid Model (TF-IDF + Optimized TF-IDF):** This is a simple ensemble model. Instead of training a single Logistic Regression model on a combined feature set, it averages the predicted probabilities from the separately trained Tuned TF-IDF model (configuration 1) and the Tuned, N-gram Optimized TF-IDF model (configuration 3). The decision threshold for this averaged probability was then optimized.

The models were primarily evaluated based on standard metrics (Accuracy, Precision, Recall, F1-score, ROC AUC). Crucially, **Precision-Recall curves** were analyzed for each model's predicted probabilities, and **decision thresholds were optimized to achieve a target recall level** (approximately 70%). This optimization aligns the model's decision-making with the project's goal of prioritizing candidates and minimizing false negatives. Confusion matrices were visualized at these optimized thresholds to understand the specific performance trade-offs in terms of true positives, false positives, true negatives, and false negatives. The notebook has no separate train and test split.

**Methodology for Prediction (How the Model Makes a Prediction):**

For a new applicant, the prediction process involves: applying the *same* preprocessing and feature engineering steps (using the fitted transformers from the chosen trained pipeline, e.g., the Optimized TF-IDF pipeline), scaling the features with the *same* StandardScaler, feeding the scaled features to the trained Logistic Regression model to get a probability score, and finally applying the **optimized decision threshold** (derived during evaluation) to classify the applicant as 'Best Match' (1) or 'Not Best Match' (0). For the Weighted Hybrid model, this would involve getting probabilities from its two constituent models and averaging them before applying the optimized threshold.

**Responsible AI Considerations:**

I removed Age, Gender, Race1 and Race2 before training. The model still uses Ethnicity, as described above. Using Ethnicity to screen candidates is not acceptable for real hiring. I plan to retrain the model without Ethnicity, Race1 and Race2 and report the new results. Logistic Regression's relative interpretability was also a factor. Further steps for a production system would include bias audits, fairness metrics, human oversight, and transparency.

### Why I am removing protected attributes from the model

I decided that protected characteristics must not be inputs to a screening model. This covers ethnicity, race, national origin, age and gender. This is a place where human judgment should override a model. A model can find a statistical signal in any column. That signal does not make the column a fair basis for a hiring decision. The current saved model makes Ethnicity, represented by national origin labels, a factor in candidate scoring. Removing Ethnicity requires a retrain, and that retrain has not happened yet.

## Results

The models were evaluated at decision thresholds optimized to achieve approximately 70% recall. The key results are summarized below.

**These are scores on the training data, not on held out data.** For the first five models, I fit each pipeline on the full oversampled dataset and scored it on the same rows. These scores are optimistic. They need a rerun with a held out split made before oversampling. They must also be measured again after retraining without Ethnicity.

*   **Tuned TF-IDF Model:** Precision: 0.6205, Recall: 0.7006, F1-score: 0.6581, ROC AUC: 0.6866.
*   **Tuned Word2Vec Model:** Precision: 0.5550, Recall: 0.7000, F1-score: 0.6191, ROC AUC: 0.6039.
*   **Hybrid Model (Default TF-IDF + Word2Vec):** Precision: 0.6286, Recall: 0.7000, F1-score: 0.6624, ROC AUC: 0.7002.
*   **Optimized TF-IDF Model (Tuned N-grams):** Precision: 0.6627, Recall: 0.7000, F1-score: 0.6808, ROC AUC: 0.7345.
*   **Optimized Hybrid Model (Optimized TF-IDF + Word2Vec):** Precision: 0.6670, Recall: 0.7002, F1-score: 0.6832, ROC AUC: 0.7450.
*   **Weighted Hybrid Model (TF-IDF + Optimized TF-IDF):** Precision: 0.5528, Recall: 0.7000, F1-score: 0.6178, ROC AUC: 0.5936.

The Weighted Hybrid score uses cross validated probabilities. The other five scores use predictions on the training rows. The two kinds of scores are not directly comparable.

With 5 fold cross validation at the default threshold, the Optimized TF-IDF model had ROC AUC 0.60, precision 0.56 and recall 0.59. These cross validation scores are also likely optimistic, because the folds share duplicate rows from oversampling.

On the training data scores, the **Optimized TF-IDF Model (Tuned N-grams)** and the **Optimized Hybrid Model (Optimized TF-IDF + Word2Vec)** had the highest precision, F1-scores, and ROC AUCs at the target recall. These rankings need a rerun with a held out split before they can support a model choice.

I chose the **Optimized TF-IDF Model (Tuned N-grams)** for the saved pipeline because it is simpler than the hybrid model. I did not measure inference time or cost.

## Saved Artifacts

The following artifacts from the modeling process have been saved:

*   `upsampled_training_data_July 21.csv`: The final upsampled and preprocessed dataset used for training. It has the columns Ethnicity, Resume, Job Roles, Job Description and Best Match.
*   `optimized_tfidf_logistic_regression_pipeline.joblib`: The trained scikit-learn Pipeline object for the **Optimized TF-IDF Model (Tuned N-grams)**. This pipeline includes the preprocessor (with optimized TF-IDF vectorizers and OneHotEncoder), the scaler, and the Logistic Regression model. The OneHotEncoder encodes Job Roles and Ethnicity. The current pipeline expects an Ethnicity column. Do not use it to score real candidates.

## Inference

For detailed information on how to use the saved model pipeline for making predictions on new data, including understanding the prediction logic and the example `predict.py` script, please refer to **Section 7: Prediction Logic and Script for Inference** within the main notebook (`TalentSol_Supervised_Classifier_for_Initial_Candidate_Screening_Public.ipynb`).

A brief overview of the prediction process:

To use the saved pipeline for making predictions on new data, you first need to load the pipeline file using `joblib.load()`. New applicant data needs to be prepared in a pandas DataFrame with the expected columns. This DataFrame is then passed to the loaded pipeline's `.predict_proba()` method to get probability scores. Finally, an optimized decision threshold (determined during model evaluation) is applied to these probabilities to classify applicants as 'Best Match' or 'Not Best Match'.

The `predict.py` script is provided in the notebook as a standalone example of how to load the pipeline and perform inference, including handling command-line input for data files. There is no separate `predict.py` file in this repository.

## Engineering Integration Recommendations

For integrating this model into a production ATS (like TalentSol), consider these principles and recommended services. These recommendations apply to a retrained model without protected attributes, not to the current saved pipeline.

*   **Start Simple and Iterate:** Begin with a straightforward and lightweight architecture. This is a key engineering tenet. For a problem like initial candidate screening, which involves handling a high volume of candidates and prioritizing for high recall, starting simple allows for faster experimentation and testing to quickly determine if the solution is effective before adding complexity. Avoid unnecessary complexity ("bloat") that could negatively impact inference performance or operational costs. Iterate and add complexity only when necessary to meet specific requirements.
*   **Explainability and Transparency Document:** Create a clear and easy-to-understand document explaining how the model works, its limitations, the data used, and how predictions are made. This document is crucial for users of the system (recruiters, HR personnel) and other stakeholders (business leads, legal) to build trust, ensure fair usage, and understand the model's impact on the hiring process.
*   **Real-time Data Pipeline:** Implement an efficient pipeline to preprocess new applicant data using the loaded pipeline's transformers. Recommended services for building data pipelines include serverless functions (e.g., Cloud Functions, Lambda), containerization platforms (e.g., Cloud Run, Kubernetes), or batch processing services (e.g., Dataflow, EMR) depending on latency and throughput requirements.
*   **Data and Model Storage:** Securely store raw data, processed data, and the trained model pipeline. Cloud Object Storage (e.g., Cloud Storage, S3, Blob Storage) is suitable for raw data and model files. Managed databases (e.g., Cloud SQL, RDS, Azure SQL) can store structured applicant data. Consider a Feature Store for managing and serving features consistently.
*   **Latency:** Monitor and optimize prediction latency. I did not benchmark latency for any model in this project.
*   **Memory:** Be mindful of the memory footprint of the loaded pipeline, especially for the TF-IDF vectorizer's vocabulary (which can be large for n-grams).
*   **Vector Databases:** For models utilizing dense vector embeddings (like the Word2Vec and Hybrid models), consider the use of a dedicated vector database. This can be essential for efficient storage, indexing, and retrieval of these embeddings at scale, particularly if implementing features beyond simple classification, such as semantic search or candidate similarity matching. While the selected Optimized TF-IDF model generates sparse features and may not require a vector database for core prediction, understanding the architectural implications of embedding-based models is valuable for evaluating the engineering complexity and cost trade-offs.
*   **Scalability:** Deploy the model as a scalable service.
*   **Monitoring:** Implement monitoring for performance tracking (e.g., inference time, error rates), data drift (changes in input data characteristics), and model drift (degradation in model performance over time).
*   **API Endpoint:** Provide a clear and well-documented API for seamless integration with other components of the ATS.
*   **Security:** Ensure secure handling and storage of sensitive applicant data and the model artifact.
*   **Versioning:** Use a system for model versioning to manage updates and rollbacks effectively.

## Web Framework/API Integration (Conceptual)

Integrating the trained model into a web framework (like Flask or FastAPI) for a prediction API endpoint involves using the core prediction logic within the web application's request handling. The web application would receive new applicant data (e.g., via a JSON payload), load the saved pipeline (ideally once when the application starts for efficiency), prepare the data in a pandas DataFrame, call the `predict_applicant` function (or equivalent logic) with the DataFrame and the optimized threshold, and return the predictions (e.g., as a JSON response). Key considerations for a production deployment include handling multiple requests concurrently, error handling, scaling, and model versioning. The notebook discusses this briefly at the end of Section 7.

## Model Risk: Keyword Matching

This section previously gave applicants tips for raising their score. I replaced it because the tips described how to game the model. The TF-IDF model weights terms that are frequent in one document and rare across others. Resumes that copy words and phrases from the job description can score higher, whether or not the candidate fits the role better. The model also scores the Ethnicity column, which no candidate should be scored on. A recruiter should treat the score as one signal for ordering review. The score should not decide the outcome for a candidate.

## Fairness

**What was tested**

- Precision, recall, F1 and ROC AUC at tuned thresholds for six configurations. Five of these scores come from the training rows. The Weighted Hybrid score comes from cross validated probabilities.
- Cross validation at the default threshold, on oversampled data that contains duplicate rows.
- Removal of Age, Gender, Race1 and Race2 before training.

**What was not tested**

- Any evaluation on a held out test split. The notebook has none.
- Selection rates by group, or adverse impact ratios.
- Error rates or calibration by group.
- Proxy features in resume text or job roles that could stand in for a protected characteristic.
- Performance on real applicant data.

**What a production system would need**

- Retraining without protected attributes, with a test set split off before any oversampling.
- An independent bias audit before any use with real candidates.
- Human review of every model recommendation. The model should help a recruiter order the work. It should not reject a candidate on its own.
- A way for candidates to ask for a human review of their application.

## Limitations

- **Ethnicity input.** The saved model uses Ethnicity. It is not acceptable for real hiring until it is retrained without Ethnicity.
- **No held out evaluation.** The reported threshold metrics are scores on the training data. They need a rerun with a held out split.
- **Duplicate rows in cross validation.** I oversampled before cross validation, so the cross validation scores are likely optimistic.
- **Bootstrap sample.** Oversampling with replacement left 1,740 of the original positive rows out of the training data.
- **Dataset.** The resume text follows a fixed template. I have not confirmed whether the Kaggle dataset is synthetic or which license applies to it. The results may not carry over to real resumes.
- **No latency or cost measurement.** The model choice is not based on measured speed or cost.
- **No group fairness testing.** See [Fairness](#fairness).

Refer to the code notebook for detailed implementation of the preprocessing, modeling, and evaluation steps.

## Corrections

The following statements were corrected on **November 20, 2026** after a check against the notebook outputs, the saved model file and the committed training CSV. The model, the notebook and the data are unchanged. Only the README was wrong or incomplete.

| Item | Previously stated | Corrected to | Cause |
|------|-------------------|--------------|-------|
| Ethnicity | "kept as a feature to support diversity hiring considerations" | Ethnicity is still an input. Keeping it was the wrong choice, and retraining without it is planned. | Scoring candidates on a protected characteristic does not support diversity hiring. |
| Responsible AI statement | "'Race' split handled carefully" | Age, Gender, Race1 and Race2 were removed. Ethnicity remains. | The statement was vague and left out Ethnicity. |
| Results | Presented as model performance | Labeled as scores on the training data that need a rerun with a held out split | Five models were fit and scored on the same rows. |
| Weighted Hybrid comparison | "did not outperform the individual optimized models" | Not directly comparable to the other five | The Weighted Hybrid uses cross validated probabilities. The others use training rows. |
| Cross validation | "particularly suitable for our upsampled dataset" | Likely optimistic because folds share duplicate rows | Oversampling ran before cross validation. |
| Model choice | Based on "cost efficiency of inference/runtime" and "faster inference times" | Chosen for simplicity. Speed and cost were not measured. | No latency or cost measurement in the notebook |
| Latency recommendation | Sparse TF-IDF features "generally contribute to lower latency" | Latency was not benchmarked | No measurement |
| Class imbalance | Oversampling to prevent bias toward a majority class | Classes were close to balanced (5,150 and 4,850). Oversampling with replacement duplicated some rows and left out 1,740 positive rows. | Did not describe what the code did |
| Privacy | Removing four columns would "ensure data privacy" | The columns are not direct inputs. Resume text can still contain personal information. | Overclaim |
| Training data file | `upsampled_training_data.csv` | `upsampled_training_data_July 21.csv` | Did not match the repository |
| Notebook file | `TalentSol_ATS_ML_Project.ipynb` | `TalentSol_Supervised_Classifier_for_Initial_Candidate_Screening_Public.ipynb` | Did not match the repository |
| Notebook section | "Section 7.3: Explain Web framework/API integration (Conceptual)" | Discussed briefly at the end of Section 7 | The notebook has no Section 7.3. |
| Resume Optimization Tips | Tips for applicants to raise their score | Replaced with a model risk note | The tips described how to game a model that also scores Ethnicity. |

The six model configurations, the use of 5 fold stratified cross validation and grid search, the threshold metrics for all six models, the bigram result, the 0.4996 threshold in the notebook `predict.py` example, and the pipeline components were rechecked against the notebook and the saved pipeline and are accurate as originally published.
