# FIVE-DAY TECHNICAL PROGRESS AND IMPLEMENTATION REPORT

## CardioCare Integration with CAD Hybrid System



**Project:** CardioCare – Multi-Agent Cardiovascular Risk Assessment Integration  

**Working Repository:** `cad-hybrid`  

**Branch:** `agents/cardiacore`  

**Integration Target:** `cad-agentic`  

**Project Period:** Five working days  

**Report Date:** 19 November 2026  

**Status:** Core implementation, testing and Day 5 investigation completed; final Git contribution and upstream integration pending.



---



# 1. PROJECT OVERVIEW



The objective of this project was to integrate the CardioCare application and its machine-learning models into the existing CAD Hybrid architecture.



CardioCare contains two distinct predictive tracks:



1\. **Track A – Coronary Artery Disease (CAD) Prediction:** Uses clinical and demographic information to estimate the probability of CAD.

2\. **Track B – Heart Failure Survival Prediction:** Uses clinical information from diagnosed heart failure patients to estimate survival-related outcomes.



The integration required more than simply transferring trained models. The models needed to be adapted to the existing project architecture, connected through standardized agent interfaces, supplied with canonical input fields, and validated through automated tests.



The work also involved investigating dataset integrity, identifying possible feature leakage, reviewing duplicate-group assignments, validating saved model artifacts and documenting limitations.



The overall objective was to make CardioCare available as two independent agents within the CAD Hybrid environment while keeping their outputs separate from the existing fusion voting mechanism until further validation.



# 2. PROJECT OBJECTIVES



The five-day work was organized around the following objectives:



1\. Understand the existing CardioCare application and its model architecture.

2\. Inspect the source datasets and understand the two predictive tracks.

3\. Develop and validate the Track A and Track B machine-learning pipelines.

4\. Understand the existing CAD Hybrid architecture and agent contract.

5\. Implement CardioCare CAD and heart failure agent adapters.

6\. Map CardioCare inputs to canonical system fields.

7\. Configure the new agents through YAML configuration files.

8\. Generate and verify model artifacts and metadata.

9\. Execute targeted and complete automated tests.

10\. Investigate possible feature leakage and ECG dataset split integrity.

11\. Prepare technical documentation and integration handover material.



# 3. DAY 1 – PROJECT UNDERSTANDING, REPOSITORY SETUP AND SOURCE INSPECTION



## 3.1 Objective



The first day focused on understanding the CardioCare project, establishing the working environment and identifying how the existing application and datasets could be integrated into the CAD Hybrid project.



The initial priority was to understand the existing implementation before making modifications.



## 3.2 Repository and branch preparation



The working repository was:



`C:\\Users\\Mamatha\\Arohan_internship\\cad-hybrid`



The working branch was established as:



`agents/cardiacore`



The branch was created from the teammate's main branch at commit:



`f58e3ba`



The purpose of creating a separate branch was to isolate CardioCare integration work from the existing project implementation.



The following principles were maintained:



- Existing ECG model work was not to be modified unnecessarily.

- The CardioCare implementation was developed in the designated working repository.

- Existing dataset split assignments were to be preserved.

- Changes intended for the separate `cad-agentic` repository were deferred until the five-day work was complete.



## 3.3 CardioCare source inspection



The original CardioCare application was located in the sibling project directory:



`..\\cardiocare-app`



The following source files were inspected:



- `app.py`

- `extention_of_Z-Alizadeh_sani_dataset.xlsx`

- `heart_failure_clinical_records_dataset.csv`

- `requirements.txt`

- `.streamlit/config.toml`



The original application was approximately 986 lines long.



The review focused on understanding:



- How patient inputs were collected.

- How the original datasets were loaded.

- How clinical features were encoded.

- How the CAD prediction workflow operated.

- How the heart failure survival model operated.

- Which model features were used.

- How predictions were presented in the original application.



The original application was treated as a reference implementation for the integration.



## 3.4 Understanding the two predictive tracks



The source application contained two different prediction tasks.



### Track A – CAD prediction



The Z-Alizadeh Sani dataset contained:



| Category | Records |

|---|---:|

| CAD positive | 216 |

| Normal | 87 |

| Total | 303 |



The target variable was `Cath`.



The task was to estimate CAD probability using clinical and demographic features.



### Track B – Heart failure survival



The heart failure clinical records dataset contained 299 patients.



Unlike Track A, this was a time-to-event prediction task.



The objective was to estimate survival-related outcomes using patient clinical information.



The two tracks were kept separate because their target definitions and prediction meanings are different.



## 3.5 Initial technical understanding



The initial inspection established the need for:



- Separate model training pipelines.

- Separate saved artifacts.

- Separate prediction functions.

- Independent feature mappings.

- Distinct agent output targets.

- Appropriate validation for each prediction task.



The investigation also established that the original CardioCare dataset represented an Iranian cohort, which would need to be documented as a limitation during integration.



## 3.6 Day 1 outcome



The repository and branch setup was completed.



The original CardioCare application and datasets were inspected, and the architecture of both predictive tracks was understood sufficiently to begin implementation.



**Deliverable:** Repository setup and initial CardioCare architecture understanding.



---



# 4. DAY 2 – MODEL DEVELOPMENT, TRAINING PIPELINE AND ARTIFACT GENERATION



## 4.1 Objective



The second day focused on understanding and implementing the standalone CardioCare machine-learning workflow within the CAD Hybrid working repository.



The work involved examining the model structure, candidate algorithms, preprocessing, training pipeline and expected model outputs.



## 4.2 Model architecture review



The CardioCare model implementation was organized in:



`cardiocare/model.py`



The implementation included the following major functions:



- `build_candidate_models()`

- `run_model_bakeoff()`

- `_prepare_track_a()`

- `train_a()`

- `predict_a()`

- `_expand_to_bins()`

- `train_b()`

- `predict_b()`



The two tracks were implemented as separate pipelines.



## 4.3 Track A – CAD model development



The Track A model used 18 clinical and demographic features.



The selected feature set included:



| No. | Feature |

|---:|---|

| 1 | Age |

| 2 | Sex |

| 3 | BMI |

| 4 | DM |

| 5 | HTN |

| 6 | Current Smoker |

| 7 | FH |

| 8 | Typical Chest Pain |

| 9 | Atypical |

| 10 | Dyspnea |

| 11 | BP |

| 12 | PR |

| 13 | LDL |

| 14 | HDL |

| 15 | TG |

| 16 | HB |

| 17 | ESR |

| 18 | EF-TTE |



The preprocessing workflow handled the original categorical values and converted the target variable into binary form.



The following candidate algorithms were considered:



- Logistic Regression

- Random Forest

- Gradient Boosting

- RBF Support Vector Classifier

- K-Nearest Neighbors



A soft-voting ensemble was also evaluated.



### Selected model



The selected Track A model was an ensemble of:



- Random Forest

- RBF Support Vector Machine



The model was evaluated using a held-out test set.



The initial reported evaluation was:



| Metric | Initial result |

|---|---:|

| Test ROC-AUC | 0.8718 |

| Brier score | 0.0988 |

| Bootstrap confidence interval | 0.7133–0.9767 |



The model was also evaluated using cross-validation and bootstrap confidence intervals.



## 4.4 Track B – Heart failure survival model development



Track B used the heart failure clinical records dataset containing 299 patients.



The model used nine features:



1\. Age

2\. Sex

3\. Anaemia

4\. Diabetes

5\. High blood pressure

6\. Smoking

7\. Ejection fraction

8\. Serum creatinine

9\. Serum sodium



The dataset contained time-to-event information.



The implementation expanded follow-up information into 30-day intervals, with up to ten intervals per patient.



An important methodological decision was to split patients before interval expansion.



This prevented intervals belonging to the same patient from being distributed between training and testing.



Grouped cross-validation and patient-cluster bootstrap evaluation were also used.



### Selected model



The selected Track B model was an ensemble of:



- Random Forest

- Logistic Regression



The split contained:



| Split | Patients |

|---|---:|

| Training | 254 |

| Testing | 45 |

| Total | 299 |



Initial evaluation:



| Metric | Result |

|---|---:|

| Interval-level test ROC-AUC | 0.7266 |

| Bootstrap confidence interval | 0.5811–0.8460 |

| Bootstrap samples | 2,000 |



The reported ROC-AUC is interval-level classification performance. It is not a C-index.



## 4.5 Training pipeline



The training process was organized through:



`scripts/train_cardiocare.py`



The script trained both tracks and generated their corresponding artifacts.



The implementation also recorded model metadata and feature-importance information.



The training process was checked through syntax validation and execution.



## 4.6 Model artifacts



The following artifacts were generated:



- `models/agents/cardiocare/cardiocare_track_a.joblib`

- `models/agents/cardiocare/cardiocare_track_b.joblib`

- `models/agents/cardiocare/metadata.json`



These files separated the trained model artifacts from the source implementation.



The metadata included model evaluation information, training prevalence and other relevant model details.



## 4.7 Day 2 outcome



Both predictive tracks were implemented and trained.



The candidate-model evaluation was completed, selected models were identified, and saved artifacts were generated.



**Deliverable:** Track A and Track B training pipelines, model artifacts and evaluation metadata.



---



# 5. DAY 3 – CAD HYBRID ARCHITECTURE, AGENT CONTRACT AND INTEGRATION DESIGN



## 5.1 Objective



The third day focused on understanding how CardioCare could fit into the existing CAD Hybrid architecture.



The objective was to avoid creating a separate prediction workflow that would not follow the existing project's agent interface.



## 5.2 Existing architecture inspection



The following project components were inspected:



- `src/cad_hybrid/fusion/fuse.py`

- `src/cad_hybrid/tabular/artifact.py`

- `src/cad_hybrid/predict.py`

- `configs/`

- `pyproject.toml`



The project structure and package organization were reviewed.



The `pyproject.toml` configuration was inspected to understand the setuptools-based source layout and test organization.



## 5.3 Fusion architecture understanding



The existing fusion module was reviewed to understand how predictions from different sources were combined.



The main integration consideration was that different agents may produce different kinds of predictions.



For example:



- An ECG agent may produce a CAD-related probability.

- A laboratory agent may produce a cardiovascular-risk proxy.

- A PPG agent may produce a non-voting evidence output.

- CardioCare Track A produces CAD probability.

- CardioCare Track B produces heart failure survival-related outputs.



These outputs cannot automatically be treated as equivalent probabilities.



The target definitions, evidence groups and training priors must be considered before fusion.



## 5.4 Training prevalence and probability interpretation



The Track A training prior was reviewed.



The training prior was calculated as:



`training_prior = y_train.mean()`



The previously inspected value was approximately 0.7121.



The training prior represents the proportion of CAD-positive cases in the training data.



It is important because the fusion system may use prior-aware probability handling.



The Track B model was treated separately because it predicts heart failure survival-related outcomes rather than CAD probability.



## 5.5 Agent contract review



The existing agent contract was reviewed in preparation for the CardioCare adapter.



The integration design required agents to return standardized information, including:



- Agent identity.

- Prediction target.

- Probability where applicable.

- Prediction label.

- Training prevalence or prior where applicable.

- Quality checks.

- Evidence or explanation.

- Model provenance.

- Relevant limitations and caveats.



This allowed CardioCare to participate in the existing architecture without requiring the fusion system to understand the original standalone application's internal implementation.



## 5.6 Integration mapping design



Two separate agent integrations were planned:



| Agent | Model track | Target |

|---|---|---|

| `cardiocare_cad` | Track A | `cad` |

| `cardiocare_hf` | Track B | `hf_survival` |



The CAD agent would return CAD probability and training prior.



The HF agent would return survival-related outputs and would not be treated as a CAD probability source.



## 5.7 Day 3 outcome



The existing architecture and fusion mechanism were reviewed.



The CardioCare agent integration design was established, including separate targets, standardized outputs and canonical feature mapping requirements.



**Deliverable:** Agent integration design and architecture compatibility review.



---



# 6. DAY 4 – AGENT ADAPTER IMPLEMENTATION, CONFIGURATION AND TESTING



## 6.1 Objective



The fourth day focused on implementing the CardioCare agent adapters and connecting them to the existing CAD Hybrid architecture.



The work also involved configuration updates, model loading, feature validation and automated testing.



## 6.2 CardioCare agent adapter



The adapter was implemented in:



`src/cad_hybrid/agents/cardiocare.py`



Two independent agents were implemented.



### 6.2.1 CardioCare CAD agent



Class:



`CardiocareCADAgent`



Agent identifier:



`cardiocare_cad`



Prediction target:



`cad`



The CAD agent:



1\. Receives canonical patient input fields.

2\. Validates required inputs.

3\. Maps canonical fields to Track A feature names.

4\. Loads the saved Track A model artifact.

5\. Calls the CardioCare prediction function.

6\. Validates the returned probability and training prior.

7\. Produces a standardized agent report.

8\. Includes model provenance and limitations.

9\. Provides feature-importance information.



The adapter uses the original Track A model rather than duplicating the training logic.



### 6.2.2 CardioCare heart failure agent



Class:



`CardiocareHFAgent`



Agent identifier:



`cardiocare_hf`



Prediction target:



`hf_survival`



The HF agent uses the Track B model.



It requires a confirmed heart failure diagnosis before making a prediction.



The required clinical inputs include:



- Age

- Sex

- Anaemia

- Diabetes

- High blood pressure

- Smoking

- Ejection fraction

- Serum creatinine

- Serum sodium



The adapter checks that the survival outputs are finite and within valid probability ranges.



It also checks that predicted survival probabilities are non-increasing over time.



The output includes:



- Predicted days.

- Survival probability.

- 90-day risk.

- 300-day risk.



The HF agent does not return a CAD probability or training prior.



## 6.3 Canonical field mapping



The configuration file:



`configs/fields.yaml`



was reviewed and used for canonical input-field definitions.



The configuration contained 28 fields.



The CardioCare mappings were checked against the original model feature requirements.



This mapping layer allows the agent to receive standardized system inputs while translating them into the feature names expected by the standalone CardioCare model.



## 6.4 Agent configuration



The configuration file:



`configs/agents.yaml`



was updated to include the CardioCare agents.



The resulting configuration included five agents:



- `lab`

- `ecg`

- `ppg`

- `cardiocare_cad`

- `cardiocare_hf`



The CardioCare entries were configured as:



| Setting | CardioCare CAD | CardioCare HF |

|---|---|---|

| Target | cad | hf_survival |

| Evidence group | clinical | hf |

| Voting | False | False |

| Weight | 0.0 | 0.0 |

| Enabled | True | True |



This allowed the agents to be tested without contributing to the existing fusion score.



This was intentional because the two CardioCare targets require separate validation and should not be automatically combined with the existing CAD probability sources.



## 6.5 Model artifact integration



The agent adapters were connected to the saved model artifacts.



The artifacts were stored under:



`models/agents/cardiocare/`



The adapter was reviewed to ensure that it loaded the appropriate model for each prediction task.



## 6.6 Testing



Targeted CardioCare tests were executed.



Result:



**12 tests passed.**



The complete project test suite was also executed.



The final recorded result was:



| Test result | Count |

|---|---:|

| Passed | 106 |

| Failed | 0 |

| Warnings | 3 |



The three warnings were non-blocking SHAP deprecation warnings.



The registry and agent loading were also checked to ensure that the five configured agents were recognized.



The command `git diff --check` completed without reporting whitespace errors during the recorded review.



## 6.7 Day 4 outcome



The CardioCare CAD and HF adapters were implemented.



Canonical field mappings and agent configuration were reviewed.



The targeted and full test suites passed.



**Deliverable:** Two CardioCare agent adapters, YAML configuration updates and automated test verification.



---



# 7. DAY 5 – CALIBRATION REVIEW, DATA LEAKAGE INVESTIGATION AND DATASET VERIFICATION



## 7.1 Objective



The fifth day focused on a deeper technical review of the model implementation, investigation of possible feature leakage, verification of ECG dataset splits and preparation of the final integration documentation.



This day included source-code inspection, exploratory statistical analysis, image comparison and verification of model artifacts.



## 7.2 Track A calibration improvement



The earlier Track A implementation used Scikit-learn's `CalibratedClassifierCV`.



The updated implementation introduced an explicit out-of-fold Platt calibration process.



The updated workflow:



1\. Divides the training data into five stratified folds.

2\. Generates out-of-fold predictions.

3\. Collects one out-of-fold prediction per training record.

4\. Converts probabilities into logits.

5\. Fits a logistic regression calibrator.

6\. Refits the selected estimator using all training data.

7\. Applies the fitted calibrator to prediction probabilities.



The relevant functions included:



- `_probability_to_logit()`

- `_fit_oof_platt_calibrator()`



Probability clipping was included to avoid numerical problems during logit conversion.



### Updated Track A results



| Metric | Updated result |

|---|---:|

| Test ROC-AUC | 0.8648 |

| Brier score | 0.0973 |

| Bootstrap confidence interval | 0.6992–0.9721 |

| Bootstrap samples | 2,000 |

| Calibration method | OOF Platt |



The AUC decreased slightly compared with the earlier result, while the Brier score improved slightly.



The change should therefore be interpreted as a calibration-method update, not as an improvement in every evaluation metric.



## 7.3 Model artifact debugging



An issue was identified in the Track A model bundle.



The `predict_a()` function required a calibrator, but the training script did not originally include the calibrator in the saved bundle.



The problem was corrected in:



`scripts/train_cardiocare.py`



The bundle was updated to include:



`"calibrator": result_a\["calibrator"]`



The training script was executed again.



The saved model bundle was inspected.



The following keys were confirmed:



- `model`

- `calibrator`

- `features`

- `training_prior`

- `global_importance`

- `importance_mode`



The calibrator was confirmed to be present.



This resolved the mismatch between model training and prediction-time requirements.



## 7.4 CardioCare feature leakage investigation



The original Z-Alizadeh Sani dataset was examined to identify additional columns that could introduce target leakage.



The 18 selected Track A features were compared against the full dataset structure.



Additional fields examined included:



- LAD

- LCX

- RCA

- Region RWMA

- VHD

- Q Wave

- St Elevation

- St Depression

- Tinversion

- LVH

- Poor R Progression

- BBB



LAD, LCX and RCA showed particularly strong relationships with the Cath target.



Region RWMA also showed a strong association.



These fields were excluded from the 18 selected model features.



## 7.5 Single-feature AUROC investigation



Individual AUROC values were calculated for all 18 selected features.



Typical Chest Pain showed an AUROC of approximately 0.7990.



Age showed an AUROC of approximately 0.7239.



No selected feature exceeded an AUROC of 0.80.



The values were calculated using all 303 records and were treated as exploratory feature-screening results, not independent test performance.



Direction-adjusted AUROC was also calculated using:



`max(AUROC, 1 - AUROC)`



This helped examine feature association strength regardless of direction.



## 7.6 Temporal leakage review



The original application was examined to understand the collection of EF-TTE.



The application labels the field as:



`Ejection Fraction, Echocardiogram (%)`



However, the original source did not establish the exact measurement timing relative to angiography or diagnosis.



This was documented as an unresolved temporal leakage question.



## 7.7 Original ECG duplicate-group investigation



The original ECG project was inspected to understand how duplicate groups and train-validation-test assignments were created.



The following source files were reviewed:



- `batch_preprocessing.py`

- `create_dataset_splits.py`

- `audit_duplicates.py`

- `verify_no_data_leakage.py`

- `ecg_dataset.py`



The investigation established that the original splitting workflow used duplicate-group identifiers.



The original `create_dataset_splits.py` used `StratifiedGroupKFold`.



The test split was selected from seven folds, targeting approximately 15%.



The validation split was selected from five folds using the remaining data.



Explicit group-overlap checks were also present.



## 7.8 SHA-256 duplicate-group consistency checks



The original `exact_duplicate_groups.csv` file was examined.



Results:



| Measurement | Result |

|---|---:|

| Total records | 695 |

| Unique duplicate groups | 258 |

| Unique SHA-256 hashes | 258 |

| Missing group IDs | 0 |

| Groups containing multiple SHA-256 hashes | 0 |

| SHA-256 hashes assigned to multiple groups | 0 |



Group-size distribution:



| Group size | Number of groups |

|---:|---:|

| 2 | 228 |

| 7 | 1 |

| 8 | 29 |



The checks established internal consistency between the recorded group IDs and SHA-256 hashes.



The exact original generation process for this CSV was not located and remains a documentation limitation.



## 7.9 ECG split verification



The current ECG split assignments were verified using:



`python scripts/verify_splits.py`



The results were:



| Split | Images |

|---|---:|

| Training | 638 |

| Validation | 158 |

| Testing | 132 |

| Total | 928 |



All 928 image paths were unique.



Duplicate-group overlap results:



| Comparison | Overlapping groups |

|---|---:|

| Training–Validation | 0 |

| Training–Testing | 0 |

| Validation–Testing | 0 |



Duplicate-group label consistency also passed.



The verification generated:



- `reports/split_manifest.csv`

- `reports/split_verification.txt`



**Result: Split verification passed.**



## 7.10 Additional pHash analysis



A separate diagnostic investigation was conducted using:



`scripts/clean_phash_splits.py`



The investigation used a pHash Hamming-distance threshold of 4.



Twenty cross-split candidate pairs were identified.



| Split combination | Candidates |

|---|---:|

| Training–Validation | 14 |

| Training–Testing | 4 |

| Validation–Testing | 2 |

| Total | 20 |



The distance distribution was:



| Distance | Candidates |

|---:|---:|

| 2 | 5 |

| 4 | 15 |



Five distance-2 pairs were selected for detailed visual inspection.



All ten original images were checked for existence.



The visual comparison focused on waveform structure, amplitudes, report layout and image similarity.



No exact duplicate was confirmed through the visual screening.



The images shared some report formatting, but their visible waveform structures differed.



The investigation also established that pHash similarity does not prove patient identity.



## 7.11 pHash cleaning script review



The `clean_phash_splits.py` implementation was inspected.



It:



- Reads existing split CSV files.

- Calculates image hashes.

- Compares images across different splits.

- Uses same-class cross-split comparisons.

- Identifies candidate matches using distance 4.

- Marks validation and test candidates for exclusion.

- Preserves training images.

- Writes to a separate cleaned output directory.

- Does not overwrite the original splits.



The script's `--write` option was not used during this investigation.



Therefore, the original split files remained unchanged.



## 7.12 Agent and configuration review



The CardioCare adapter was reviewed again.



The CAD agent was confirmed to use Track A and return CAD probability.



The HF agent was confirmed to use Track B and return survival-related outputs.



The configuration files were checked:



- `configs/agents.yaml`

- `configs/fields.yaml`



The validation confirmed:



- Five configured agents.

- Twenty-eight canonical fields.

- Both CardioCare agents present.

- Required CardioCare field mappings available.

- Correct prediction targets.

- CardioCare agents enabled for testing.

- Voting disabled and weights set to zero.



No additional configuration changes were required during this review.



## 7.13 Final test execution



The targeted CardioCare tests were executed:



**12 passed.**



The complete project test suite was executed:



**106 passed, 3 warnings.**



The warnings were non-blocking SHAP `PendingDeprecationWarning` messages.



The modified source files were reviewed.



The command:



`git diff --check`



completed without reporting whitespace errors.



## 7.14 Day 5 outcome



The model calibration implementation was reviewed and corrected.



The saved model bundle was verified.



The CardioCare feature leakage investigation, original ECG duplicate-group review, split verification and pHash visual analysis were completed.



The agent configurations and complete test suite were verified.



The detailed report was saved as:



`reports/cardiocare_leakage_report.md`



**Deliverable:** Day 5 leakage investigation report, split verification evidence, pHash diagnostic evidence and updated model artifact.



---



# 8. CONSOLIDATED TECHNICAL RESULTS



## 8.1 Model results



| Evaluation | Track A | Track B |

|---|---:|---:|

| Dataset records | 303 | 299 patients |

| Selected model | RF + RBF-SVM | RF + Logistic Regression |

| Test metric | ROC-AUC | Interval-level ROC-AUC |

| Test result | 0.8648 | 0.7266 |

| Brier score | 0.0973 | Not reported |

| Bootstrap samples | 2,000 | 2,000 |

| Confidence interval | 0.6992–0.9721 | 0.5811–0.8460 |



## 8.2 Dataset verification results



| Verification | Result |

|---|---|

| ECG images checked | 928 |

| Unique image paths | 928 |

| Train images | 638 |

| Validation images | 158 |

| Test images | 132 |

| Train-validation group overlap | 0 |

| Train-test group overlap | 0 |

| Validation-test group overlap | 0 |

| Duplicate-group label consistency | Passed |

| pHash candidate pairs | 20 |

| Visually confirmed exact duplicates | None in reviewed pairs |



## 8.3 Integration and testing results



| Component | Result |

|---|---|

| Track A model | Implemented |

| Track B model | Implemented |

| Training pipeline | Implemented |

| Model artifacts | Generated |

| CAD agent adapter | Implemented |

| HF agent adapter | Implemented |

| Canonical field mappings | Reviewed |

| Agent configuration | Validated |

| Targeted tests | 12 passed |

| Full test suite | 106 passed |

| Full-suite failures | 0 |

| Warnings | 3 non-blocking |



---



# 9. IMPORTANT FILES CREATED OR MODIFIED



## 9.1 Model implementation and training



| File | Purpose |

|---|---|

| `cardiocare/model.py` | Track A and Track B model implementation, training, prediction and calibration |

| `scripts/train_cardiocare.py` | Model training and artifact generation |

| `models/agents/cardiocare/cardiocare_track_a.joblib` | Saved CAD model bundle |

| `models/agents/cardiocare/cardiocare_track_b.joblib` | Saved heart failure model bundle |

| `models/agents/cardiocare/metadata.json` | Model evaluation and metadata |



## 9.2 Agent integration and configuration



| File | Purpose |

|---|---|

| `src/cad_hybrid/agents/cardiocare.py` | CardioCare CAD and HF agent adapters |

| `configs/agents.yaml` | Agent registration and voting configuration |

| `configs/fields.yaml` | Canonical clinical field definitions |



## 9.3 Dataset verification and evidence



| File | Purpose |

|---|---|

| `scripts/verify_splits.py` | Verifies ECG split integrity |

| `scripts/clean_phash_splits.py` | Additional cross-split pHash investigation |

| `reports/split_manifest.csv` | Split assignment manifest |

| `reports/split_verification.txt` | Split verification output |

| `reports/phash_20_pairs.csv` | Candidate pHash pairs |

| `reports/phash_20_pairs.png` | Candidate-pair visual summary |

| `reports/phash_distance2_comparison.png` | Detailed comparison of five pairs |

| `reports/final_normal_pair.png` | Detailed Normal image comparison |

| `reports/cardiocare_leakage_report.md` | Day 5 investigation report |



---



# 10. TECHNICAL CHALLENGES AND RESOLUTIONS



## Challenge 1 – Integrating two different prediction tasks



**Problem:** CardioCare contains a CAD prediction model and a heart failure survival model. Their prediction targets and output meanings are different.



**Resolution:** Implemented two independent agent adapters with separate targets: `cad` and `hf_survival`.



## Challenge 2 – Mapping standalone model inputs to system fields



**Problem:** The original CardioCare model expects feature names that differ from the canonical fields in CAD Hybrid.



**Resolution:** Implemented feature mapping in the CardioCare adapter and reviewed the 28 canonical field definitions.



## Challenge 3 – Preventing incorrect fusion participation



**Problem:** The HF survival output should not be interpreted as a CAD probability, and the CardioCare CAD output should not automatically influence fusion before validation.



**Resolution:** Configured the CardioCare agents as enabled for testing but with voting disabled and weight zero.



## Challenge 4 – Missing calibrator in the saved model bundle



**Problem:** The prediction function required a calibrator that was not included in the original Track A saved bundle.



**Resolution:** Updated `scripts/train_cardiocare.py` to include `result_a\["calibrator"]`, regenerated the artifact and verified the saved bundle keys.



## Challenge 5 – Calibration implementation review



**Problem:** The calibration approach needed to produce out-of-fold predictions for fitting the Platt calibrator.



**Resolution:** Implemented explicit five-fold OOF Platt calibration, including probability-to-logit conversion and prediction-time calibration.



## Challenge 6 – Possible feature leakage



**Problem:** The original CAD dataset contained additional angiographic fields strongly associated with the target.



**Resolution:** Inspected the additional dataset columns and verified that LAD, LCX, RCA and Region RWMA were excluded from the selected 18 model inputs.



## Challenge 7 – ECG duplicate-group integrity



**Problem:** Related or duplicate ECG images across splits could lead to overoptimistic model evaluation.



**Resolution:** Traced the original group-aware splitting workflow, inspected SHA-256 consistency and ran split verification.



## Challenge 8 – Suspicious pHash matches



**Problem:** Twenty cross-split candidate pairs were identified using pHash.



**Resolution:** Verified image existence and visually inspected the distance-2 candidates. No exact duplicates were confirmed among the reviewed pairs. The original split files were preserved.



---



# 11. LIMITATIONS AND TECHNICAL CONSIDERATIONS



The following limitations were documented:



1\. Patient-level independence cannot be established using duplicate-group IDs alone.

2\. The exact generation process of `exact_duplicate_groups.csv` was not located.

3\. pHash similarity is not proof of duplicate patient records.

4\. The suspicious pHash candidates require further waveform-specific analysis if stronger evidence is needed.

5\. The timing of EF-TTE and other clinical measurements relative to diagnosis requires confirmation.

6\. Track A uses an Iranian angiography-referred cohort.

7\. Track B uses a separate 299-patient heart failure dataset.

8\. External validation is required before making broader performance claims.

9\. Calibration performance must be evaluated on independent data.

10\. CardioCare agents remain excluded from fusion voting until further review.



These limitations are important for responsible reporting of the research prototype.



---



# 12. TIME AND EFFORT RECORD



The available project discussions and terminal outputs establish the work performed, but exact start and end times for every task were not recorded.



Therefore, exact daily hours and total hours are marked as not recorded.



| Day | Main work | Recorded working hours |

|---|---|---|

| Day 1 | Repository setup, source inspection and architecture understanding | Not recorded |

| Day 2 | Model development, training and artifact generation | Not recorded |

| Day 3 | CAD Hybrid architecture and agent integration design | Not recorded |

| Day 4 | Agent adapters, configuration and automated testing | Not recorded |

| Day 5 | Calibration review, leakage investigation and split verification | Not recorded |

| **Total** | **Five-day technical work** | **Not recorded** |



This is preferable to estimating hours without supporting records.



---



# 13. CURRENT GIT AND CONTRIBUTION STATUS



The CardioCare work was developed in the `cad-hybrid` working repository on the `agents/cardiacore` branch.



The latest reviewed changes included:



- `cardiocare/model.py`

- `scripts/train_cardiocare.py`

- `models/agents/cardiocare/metadata.json`



Additional local investigation and backup files were retained for review.



The following actions remain pending:



1\. Final review of all modified and untracked files.

2\. Decide which diagnostic evidence files should be included in the contribution.

3\. Exclude temporary backup files from the contribution.

4\. Organize the five-day work into meaningful commits.

5\. Create the intended contribution branch in Jay's `cad-agentic` repository after the work is finalized.

6\. Check write permissions or use a fork if required.

7\. Push the contribution branch.

8\. Prepare a pull request.

9\. Complete the final integration review.



**Important:** No final CardioCare commit, push or pull request is claimed as completed in this report.



The original `ecg-model` project and the separate `cad-agentic` repository were not modified during this CardioCare work.



---



# 14. FINAL FIVE-DAY COMPLETION SUMMARY



| Workstream | Status |

|---|---|

| Repository and environment preparation | Completed |

| CardioCare source inspection | Completed |

| Track A model development | Completed |

| Track B model development | Completed |

| Model training and artifact generation | Completed |

| CAD Hybrid architecture review | Completed |

| Agent contract review | Completed |

| CAD agent adapter | Completed |

| HF agent adapter | Completed |

| Canonical field mapping | Completed |

| Agent YAML configuration | Completed |

| OOF Platt calibration update | Completed |

| Saved calibrator verification | Passed |

| Feature leakage investigation | Completed |

| ECG duplicate-group investigation | Completed |

| SHA-256 consistency verification | Passed |

| ECG split verification | Passed |

| pHash candidate analysis | Completed |

| Visual candidate inspection | Completed |

| Targeted CardioCare tests | 12 passed |

| Full project test suite | 106 passed |

| Day 5 investigation report | Saved |

| Final Git contribution | Pending |

| Upstream branch and pull request | Pending |



# 15. OVERALL CONCLUSION



The five-day CardioCare integration work established two independent predictive model tracks and connected them to the CAD Hybrid architecture through standardized agent adapters.



Track A provides CAD probability predictions, while Track B provides heart failure survival-related outputs. Their separate targets and output meanings are maintained in the integration.



The work included model development, training, evaluation, artifact generation, agent implementation, canonical feature mapping, configuration validation and automated testing.



A significant part of the final investigation involved examining data integrity and possible leakage risks. The original ECG project used group-aware stratified splitting, and the current split verification reported zero duplicate-group overlap across training, validation and testing.



The additional pHash analysis identified 20 cross-split candidate pairs. Visual inspection did not confirm exact duplicates among the reviewed pairs. Patient-level independence and complete absence of leakage cannot be claimed from the available evidence.



The Track A calibration implementation was also updated, and the saved model bundle was corrected to include the calibrator. The targeted CardioCare tests passed, and the complete project test suite finished with 106 passed tests and three non-blocking warnings.



At the end of the five-day work, the CardioCare implementation, model artifacts, agent adapters, configuration review and investigation documentation were available in the working repository.



The remaining work is to complete the final Git review, organize the contribution, prepare the branch in the separate `cad-agentic` repository and submit the pull request.



**Final status:** Core CardioCare development, integration preparation, testing and Day 5 investigation completed. Final upstream Git contribution and integration approval remain pending.



*This project is an academic research and software-integration prototype. The reported model results do not establish clinical readiness, diagnostic validity or suitability for independent patient care.*


