# Viva Preparation — Telecom Complaint Classification

## Core questions

**1. What is Machine Learning?**  
Machine Learning is a branch of AI in which a system learns patterns from data to make predictions or decisions without being explicitly programmed for every case.

**2. What is the objective of this project?**  
To classify a telecom complaint into a likely issue category using structured complaint information.

**3. Why is this classification?**  
The output is a discrete category such as a complaint issue label, not a continuous numerical value.

**4. What is the dataset source?**  
The Federal Communications Commission (FCC) CGB Consumer Complaints public dataset.

**5. What is X and y?**  
X contains service form, method, state, month, year and day of week. y is the specific complaint issue.

**6. What is preprocessing?**  
Preparing raw data for modelling by removing duplicates, handling missing values, parsing dates, encoding categorical values and scaling numerical values.

**7. Why encode categorical data?**  
Most ML algorithms require numeric inputs. One-hot encoding converts categories into numeric indicator columns without inventing an order.

**8. Why scale numerical data?**  
Scaling puts numeric features on comparable scales. It is useful for Logistic Regression optimization. The Random Forest model does not generally require scaling, but the shared pipeline keeps the process consistent.

**9. Why split the dataset?**  
Training data teaches the model; test data estimates how well it predicts unseen records.

**10. Why 80:20?**  
It provides most records for learning while retaining a separate portion for evaluation. It is a common practical split, not a universal rule.

**11. What is random_state=42?**  
It fixes the random split so the experiment can be reproduced.

**12. What is stratification?**  
It preserves approximately the same class proportions in training and testing subsets.

**13. Explain Logistic Regression.**  
It estimates the probability of each class from a weighted combination of input features. Despite its name, it is widely used for classification.

**14. Explain Random Forest.**  
It trains multiple decision trees on varied samples/features and combines their predictions to improve robustness.

**15. What is accuracy?**  
The fraction of predictions that are correct.

**16. What is precision?**  
Among records predicted as a class, the fraction that truly belong to that class.

**17. What is recall?**  
Among records that truly belong to a class, the fraction correctly identified.

**18. What is F1-score?**  
The harmonic mean of precision and recall.

**19. Why macro F1?**  
It calculates F1 for each class and averages them equally, so minority categories are not overwhelmed by the most common category.

**20. What is a confusion matrix?**  
A table comparing actual and predicted classes, showing correct classifications and the categories that are confused.

**21. What is overfitting?**  
When a model learns training-specific patterns/noise and performs poorly on unseen data.

**22. What is a pipeline?**  
A sequence of preprocessing and modelling steps bundled together so the same transformations are applied consistently.

**23. Why save the model?**  
So the frontend can load the trained model and make predictions without training again each time.

**24. What is Joblib?**  
A Python library used here to serialize and load the fitted scikit-learn pipeline.

**25. What is Streamlit?**  
A Python framework for creating interactive web applications and dashboards.

## Important project limitation

This project predicts the issue category of an already submitted FCC complaint. It does not predict whether a telecom customer will file a complaint. The FCC dataset is self-selected and is not a complete or representative sample of all telecom customers.
