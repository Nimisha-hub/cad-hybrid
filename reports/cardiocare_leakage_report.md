# DAY 5 – COMPREHENSIVE DATA LEAKAGE INVESTIGATION, MODEL VALIDATION AND INTEGRATION REVIEW



**Project:** CardioCare Integration with CAD Hybrid System  

**Repository:** `cad-hybrid`  

**Investigation Date:** 19 November 2026  

**Scope:** CardioCare Track A, Track B, ECG Dataset Integrity, Duplicate Detection, Data Splitting, Agent Configuration and Integration Readiness



---



# 1. Executive Summary



The fifth day of the CardioCare integration work focused on a detailed investigation of data leakage risks, verification of the existing ECG dataset splits, examination of the original dataset preparation workflow, and technical review of the CardioCare model and agent integration.



The objective was not simply to run the test suite. The investigation examined how the source datasets were prepared, how duplicate groups were assigned, how the original train-validation-test splits were created, whether suspicious image similarities existed across splits, and whether the CardioCare model used potentially outcome-proximal features.



The work included:



1\. Reviewing the CardioCare CAD prediction feature set.

2\. Performing exploratory single-feature AUROC analysis.

3\. Identifying additional dataset columns that could introduce target leakage.

4\. Inspecting the original CardioCare application and dataset structure.

5\. Tracing the original ECG duplicate-group and dataset-splitting workflow.

6\. Verifying SHA-256 consistency in the duplicate-group file.

7\. Reviewing the StratifiedGroupKFold-based splitting implementation.

8\. Verifying all 928 ECG image assignments.

9\. Performing additional perceptual-hash investigations across splits.

10\. Visually examining suspicious image pairs.

11\. Reviewing the CardioCare model calibration implementation.

12\. Verifying that the calibration object is included in the saved model artifact.

13\. Reviewing agent mappings and YAML configurations.

14\. Running the targeted and complete automated test suites.

15\. Reviewing the modified source files and preparing the integration handover.



The complete automated test suite passed with **106 tests passed and 3 warnings**. The warnings were non-blocking SHAP deprecation warnings.



The original duplicate-group assignments also passed the current split verification, with zero overlapping duplicate-group IDs across training, validation and testing.



However, the investigation does not establish complete patient-level independence or prove the total absence of all forms of data leakage.



# 2. Objectives of the Investigation



The Day 5 investigation was divided into five major objectives.



| Objective | Investigation |

|---|---|

| Feature leakage analysis | Identify potentially problematic input features in CardioCare Track A |

| Dataset integrity | Verify the original ECG duplicate-group assignments |

| Split verification | Check group overlap across training, validation and testing |

| Visual similarity analysis | Examine suspicious pHash matches between splits |

| Integration review | Validate model artifacts, agent mappings, configurations and tests |



The investigation also aimed to prepare evidence that could be used during technical review and integration with the separate `cad-agentic` repository.



# 3. CardioCare Track A – Feature Leakage Investigation



## 3.1 Dataset overview



Track A uses the Z-Alizadeh Sani coronary artery disease dataset.



The dataset contains 303 patient records.



| Target category | Records |

|---|---:|

| CAD positive | 216 |

| Normal | 87 |

| **Total** | **303** |



The target variable is `Cath`.



The binary target encoding is:



- CAD = 1

- Normal = 0



The dataset is an Iranian angiography-referred cohort. It should not be interpreted as a general-population dataset.



## 3.2 Selected input features



The implemented Track A model uses 18 input features.



| No. | Feature | Description |

|---:|---|---|

| 1 | Age | Patient age |

| 2 | Sex | Patient sex |

| 3 | BMI | Body Mass Index |

| 4 | DM | Diabetes mellitus |

| 5 | HTN | Hypertension |

| 6 | Current Smoker | Current smoking status |

| 7 | FH | Family history |

| 8 | Typical Chest Pain | Typical chest-pain indicator |

| 9 | Atypical | Atypical chest-pain indicator |

| 10 | Dyspnea | Breathlessness indicator |

| 11 | BP | Blood pressure |

| 12 | PR | Pulse rate |

| 13 | LDL | Low-density lipoprotein |

| 14 | HDL | High-density lipoprotein |

| 15 | TG | Triglycerides |

| 16 | HB | Hemoglobin |

| 17 | ESR | Erythrocyte sedimentation rate |

| 18 | EF-TTE | Echocardiographic ejection fraction |



The feature list was checked against the implementation in `cardiocare/model.py`.



The preprocessing workflow was also reviewed, including the mapping of binary Y/N values, sex encoding and the Cath target.



## 3.3 Inspection of additional dataset columns



The original Excel dataset contains 59 columns, including the 18 selected model features and several additional clinical and angiographic fields.



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



These additional fields were reviewed because some may contain information closely associated with the final angiographic diagnosis.



### 3.3.1 Angiographic fields



The following fields showed particularly strong relationships with the Cath target.



| LAD value | CAD | Normal |

|---|---:|---:|

| Normal | 40 | 86 |

| Stenotic | 176 | 1 |



| LCX value | CAD | Normal |

|---|---:|---:|

| Normal | 97 | 87 |

| Stenotic | 119 | 0 |



| RCA value | CAD | Normal |

|---|---:|---:|

| Normal | 102 | 87 |

| Stenotic | 114 | 0 |



These fields are closely connected to the angiographic assessment used to establish coronary artery disease status.



Including such fields in a model intended to predict CAD risk from earlier clinical information could introduce serious target leakage.



**Finding:** LAD, LCX and RCA are not included in the 18 selected model features.



### 3.3.2 Region RWMA



The Region RWMA field was also examined.



| Region RWMA | CAD | Normal |

|---|---:|---:|

| 0 | 134 | 83 |

| 1 | 23 | 3 |

| 2 | 31 | 1 |

| 3 | 14 | 0 |

| 4 | 14 | 0 |



The field showed a strong association with the target and was excluded from the model input list.



### 3.3.3 VHD



The VHD field was also inspected.



| VHD category | CAD | Normal |

|---|---:|---:|

| Moderate | 22 | 5 |

| N | 76 | 40 |

| Severe | 3 | 8 |

| Mild | 115 | 34 |



VHD was not included in the 18 selected features.



These cross-tabulations are descriptive associations. They do not independently prove leakage, but they support keeping these additional fields outside the selected input set.



## 3.4 Single-feature AUROC analysis



An exploratory single-feature AUROC calculation was performed for all 18 selected features.



The purpose was to identify features with unusually strong individual predictive relationships and to support further investigation.



| Feature | AUROC | Direction-adjusted AUROC |

|---|---:|---:|

| Age | 0.7239 | 0.7239 |

| Sex | 0.5366 | 0.5366 |

| BMI | 0.5419 | 0.4581 |

| DM | 0.6277 | 0.6277 |

| HTN | 0.6564 | 0.6564 |

| Current Smoker | 0.5330 | 0.5330 |

| FH | 0.5144 | 0.5144 |

| Typical Chest Pain | 0.7990 | 0.7990 |

| Atypical | 0.7120 | 0.2880 |

| Dyspnea | 0.5687 | 0.4313 |

| BP | 0.6614 | 0.6614 |

| PR | 0.6024 | 0.6024 |

| LDL | 0.5188 | 0.4812 |

| HDL | 0.5246 | 0.4754 |

| TG | 0.6180 | 0.6180 |

| HB | 0.5294 | 0.4706 |

| ESR | 0.6339 | 0.6339 |

| EF-TTE | 0.6870 | 0.3130 |



Direction-adjusted AUROC was calculated as:



`max(AUROC, 1 - AUROC)`



This allows the strength of a feature's relationship to be examined independently of its direction.



### Interpretation



- Typical Chest Pain had the highest individual AUROC, approximately 0.7990.

- Age also showed a notable individual association, with an AUROC of 0.7239.

- Several other features showed moderate or relatively weak individual discrimination.

- No selected feature exceeded an AUROC of 0.80.



The AUROC values were calculated using all 303 records, including records subsequently used in training and testing.



Therefore, these values are exploratory feature-screening results. They must not be presented as independent test performance or as proof that a feature is free from leakage.



## 3.5 Temporal leakage investigation



A separate concern was whether the measurements were collected before or after the diagnostic procedure.



EF-TTE was examined in particular.



The original application displays this feature as:



**Ejection Fraction, Echocardiogram (%)**



It collects a numerical value through the application interface.



However, the original source material does not establish precisely when this measurement was recorded relative to angiography or diagnosis.



The same concern applies to the timing of certain clinical measurements.



**Finding:** No confirmed temporal leakage was established, but it cannot be ruled out without patient-level source documentation and measurement timestamps.



# 4. Original ECG Dataset – Duplicate Detection Investigation



## 4.1 Purpose



The ECG dataset contains 928 images belonging to three classes:



| Class | Images |

|---|---:|

| CAD positive | 282 |

| Negative | 196 |

| Arrhythmia | 160 |

| **Total** | **928** |



Because multiple images may represent duplicate or near-duplicate records, the splitting methodology must ensure that related images assigned to a common duplicate group do not appear across training, validation and testing.



The investigation therefore examined the original dataset preparation scripts rather than relying only on the current split CSV files.



## 4.2 Review of original preprocessing



The following original project files were inspected:



- `batch_preprocessing.py`

- `create_dataset_splits.py`

- `audit_duplicates.py`

- `verify_no_data_leakage.py`

- `ecg_dataset.py`



The review established the following workflow:



1\. Duplicate information is read from `exact_duplicate_groups.csv`.

2\. The preprocessing script maps source file paths to their assigned duplicate-group IDs.

3\. Duplicate-group IDs are retained in the ECG image index.

4\. Images without an existing group assignment receive unique identifiers.

5\. The dataset splitting script uses these group identifiers during split generation.

6\. Verification checks ensure that duplicate groups do not cross split boundaries.



## 4.3 Original pHash audit



The original `audit_duplicates.py` implementation calculates perceptual hashes for ECG images.



It uses a pHash Hamming-distance threshold of 5 to identify potential visually similar pairs.



A smaller Hamming distance indicates greater similarity between the perceptual hashes.



The audit script identifies candidate pairs. The inspected implementation does not itself demonstrate how the final `exact_duplicate_groups.csv` file was generated.



This distinction is important: identifying candidate pairs and creating verified duplicate groups are separate operations.



## 4.4 SHA-256 consistency verification



The file `exact_duplicate_groups.csv` was examined using Python and Pandas.



The following checks were performed:



- Total records in the file.

- Number of unique duplicate-group IDs.

- Number of unique SHA-256 hashes.

- Missing group IDs.

- Whether a single duplicate group contains multiple SHA-256 hashes.

- Whether a SHA-256 hash appears under multiple group IDs.



### Results



| Check | Result |

|---|---:|

| Total records | 695 |

| Unique duplicate groups | 258 |

| Unique SHA-256 hashes | 258 |

| Missing group IDs | 0 |

| Groups with multiple SHA-256 hashes | 0 |

| SHA-256 hashes assigned to multiple groups | 0 |



### Group-size distribution



| Group size | Number of groups |

|---:|---:|

| 2 | 228 |

| 7 | 1 |

| 8 | 29 |



The results indicate that each recorded duplicate group is internally associated with one SHA-256 hash, and each recorded hash is associated with one group ID.



This provides evidence of internal consistency in the duplicate-group file.



However, the exact script or manual process that originally generated the CSV was not located during this investigation.



## 4.5 Original group-aware splitting methodology



The original `create_dataset_splits.py` file was inspected.



The implementation uses:



`StratifiedGroupKFold`



The grouping variable is:



`dup_group`



The splitting procedure follows two stages.



**Stage 1 – Test split**



- Seven folds are used.

- A fold closest to the target test proportion of 15% is selected.

- Class distribution and split size are considered when selecting the fold.



**Stage 2 – Validation split**



- The remaining records are used.

- Five folds are considered.

- A validation proportion close to 15% is targeted.



The script also performs explicit group-intersection checks to verify that no duplicate-group IDs are shared across the three splits.



The implementation was reviewed to establish that group IDs are supplied directly to the splitting algorithm, rather than treating every image as an unrelated observation.



## 4.6 Preservation of original assignments in cad-hybrid



The integration repository contains `split_dataset.py`.



Its documentation and implementation indicate that:



- Original `ecg-model` split CSV files are the source of truth.

- Existing duplicate-group IDs are preserved.

- Original train, validation and test assignments are retained.

- Image paths are updated for the cad-hybrid project.

- SHA-256 values are calculated and checked.

- Duplicate-group label consistency is verified.

- Cross-split duplicate-group overlap is checked.



The investigation confirmed that the integration script preserves original assignments instead of creating a completely new split from scratch.



# 5. Current ECG Split Verification



## 5.1 Split counts



The following expected split counts were checked:



| Dataset split | Expected images | Verified images |

|---|---:|---:|

| Training | 638 | 638 |

| Validation | 158 | 158 |

| Testing | 132 | 132 |

| **Total** | **928** | **928** |



All image paths were also checked for uniqueness.



The verification reported 928 unique image paths.



## 5.2 Duplicate-group overlap verification



The following comparisons were performed:



| Comparison | Overlapping groups |

|---|---:|

| Training vs Validation | 0 |

| Training vs Testing | 0 |

| Validation vs Testing | 0 |

| **Total overlapping groups** | **0** |



The duplicate-group label consistency check also passed.



### Class distribution by split



| Class | Training | Validation | Testing |

|---|---:|---:|---:|

| CAD positive | 282 | 70 | 59 |

| Negative | 196 | 48 | 40 |

| Arrhythmia | 160 | 40 | 33 |

| **Total** | **638** | **158** | **132** |



### Unique groups per class



| Split | Arrhythmia | CAD positive | Negative |

|---|---:|---:|---:|

| Training | 160 | 81 | 98 |

| Validation | 40 | 20 | 24 |

| Testing | 33 | 15 | 20 |



### Verification output



The command:



`python scripts/verify_splits.py`



reported:



- All three split counts passed.

- Total image count passed.

- Unique image path check passed.

- Duplicate-group overlap checks passed.

- Duplicate-group label consistency passed.

- Split manifest generated.

- Split verification report generated.



The verification generated:



- `reports/split_manifest.csv`

- `reports/split_verification.txt`



**Result: Task 3 – Split Verification PASSED.**



# 6. Additional Cross-Split pHash Investigation



## 6.1 Purpose



Although the existing duplicate-group checks passed, a separate pHash diagnostic was performed to examine whether visually similar images remained across different splits.



This was an additional investigation, separate from the original duplicate-group assignment process.



The script reviewed was:



`clean_phash_splits.py`



It uses a pHash Hamming-distance threshold of 4.



## 6.2 Candidate pair analysis



The investigation identified 20 cross-split candidate pairs.



| Split combination | Candidate pairs |

|---|---:|

| Training–Validation | 14 |

| Training–Testing | 4 |

| Validation–Testing | 2 |

| **Total** | **20** |



The distance distribution was:



| pHash distance | Candidate pairs |

|---:|---:|

| 2 | 5 |

| 4 | 15 |

| **Total** | **20** |



The candidate pairs were also grouped by label:



| Label | Candidate pairs |

|---|---:|

| CAD positive | 15 |

| Negative | 5 |



All 10 original images involved in the five distance-2 pairs were checked for existence.



## 6.3 Distance-2 candidate pairs



The following five pairs were identified:



| Pair | Image A | Image B | Split combination | Label | Distance |

|---:|---|---|---|---|---:|

| 1 | MI(39).jpg | MI(73).jpg | Train–Test | CAD positive | 2 |

| 2 | Normal(185).jpg | Normal(40).jpg | Train–Validation | Negative | 2 |

| 3 | PMI(137).jpg | PMI(95).jpg | Train–Validation | CAD positive | 2 |

| 4 | Normal(47).jpg | Normal(186).jpg | Train–Test | Negative | 2 |

| 5 | Normal(183).jpg | Normal(243).jpg | Train–Validation | Negative | 2 |



## 6.4 Visual inspection



The candidate pairs were visually compared using generated image-comparison outputs.



The following diagnostic files were prepared:



- `reports/phash_20_pairs.csv`

- `reports/phash_20_pairs.png`

- `reports/phash_distance2_comparison.png`

- `reports/final_normal_pair.png`



The visual inspection focused on:



- Similarity of the ECG report layout.

- Waveform shape and morphology.

- Differences in waveform amplitudes.

- Similarity of the visible report patterns.

- Whether the pair appeared to be an exact image duplicate.



The reviewed pairs showed common report formatting and some visual similarities. However, the waveform structures were visibly different in the inspected comparisons.



No exact duplicate was confirmed through this visual screening.



This does not prove that the images belong to different patients. Visual comparison alone cannot establish patient identity.



## 6.5 Important pHash interpretation



pHash is a visual similarity technique. It is not a patient-identity detection method.



ECG reports may share:



- Identical page layouts.

- Similar grid patterns.

- Repeated text and labels.

- Common formatting and background structures.



These common elements may contribute to similar perceptual hashes even when waveform content differs.



Consequently, pHash candidates should be treated as investigation leads rather than confirmed duplicate records.



## 6.6 Review of the pHash cleaning script



The `clean_phash_splits.py` implementation was examined in detail.



The script:



1\. Reads the existing train, validation and test CSV files.

2\. Loads image paths and labels.

3\. Calculates image perceptual hashes.

4\. Compares images across different splits.

5\. Considers only same-class cross-split candidate pairs.

6\. Uses a Hamming-distance threshold of 4.

7\. Marks matching validation and test candidates for exclusion.

8\. Preserves all training images.

9\. Creates a separate cleaned output directory.

10\. Verifies that direct same-class pHash matches no longer remain in the cleaned splits.



The script uses direct pairwise comparisons rather than connected-component grouping. It therefore does not establish transitive duplicate relationships.



Importantly, the script writes to:



`data/processed/phash_cleaned`



It does not overwrite the original train, validation or test files.



The `--write` option was not used during this investigation. The original split assignments remain unchanged.



### Conclusion of pHash investigation



Twenty candidate pairs were identified, and visual examination did not confirm exact duplicates among the inspected candidates.



The original duplicate-group verification remains passed. Further waveform-region-specific similarity analysis and patient-level identifiers would provide stronger evidence.



# 7. Review of the Original ECG Duplicate-Grouping Implementation



The function `group_near_duplicates()` in:



`src/cad_hybrid/data/dedupe.py`



was reviewed.



The function:



- Receives a collection of image hashes.

- Compares pairwise Hamming distances.

- Uses a configurable distance threshold.

- Maintains parent relationships.

- Merges related items into connected groups.

- Assigns a group identifier to each resulting component.



The implementation uses a union-find-style structure.



The `image_phash()` function calculates a perceptual hash using the imagehash and PIL libraries.



A separate search of the repository established that the original `split_dataset.py` uses the existing duplicate-group IDs rather than generating them again through `group_near_duplicates()`.



The original duplicate-group workflow and the additional pHash diagnostic workflow are therefore distinct and should not be confused.



# 8. CardioCare Model Calibration Review



In addition to data leakage analysis, the updated CardioCare model implementation was reviewed.



The primary source file examined was:



`cardiocare/model.py`



## 8.1 Previous calibration approach



The earlier implementation used Scikit-learn's `CalibratedClassifierCV`.



The updated implementation uses an explicit out-of-fold Platt calibration process.



## 8.2 Updated OOF Platt calibration



The updated workflow includes:



1\. Splitting the training data into five stratified folds.

2\. Generating out-of-fold predictions.

3\. Collecting one out-of-fold prediction per training record.

4\. Converting probabilities into logits.

5\. Fitting a logistic regression calibrator on the logits.

6\. Refitting the selected estimator on the full training data.

7\. Applying the calibrator to the model's probabilities.



The helper functions reviewed include:



- `_probability_to_logit`

- `_fit_oof_platt_calibrator`



Probability clipping is applied before logit conversion to avoid numerical issues at probability boundaries.



## 8.3 Track A evaluation



The updated Track A evaluation results were:



| Metric | Result |

|---|---:|

| Test ROC-AUC | 0.8648 |

| Test Brier score | 0.0973 |

| Bootstrap confidence interval | 0.6992–0.9721 |

| Bootstrap samples | 2,000 |

| Calibration method | OOF Platt |



The bootstrap confidence interval was calculated using stratified row resampling.



The earlier Track A result had a test AUC of approximately 0.8718 and Brier score of approximately 0.0988.



The updated AUC decreased slightly, while the Brier score improved slightly. The calibration change should therefore not be described as improving every metric.



## 8.4 Track B evaluation



Track B uses the heart failure clinical records dataset.



The dataset contains 299 patients.



The implementation:



- Splits patients before interval expansion.

- Expands follow-up into 30-day intervals.

- Uses grouped cross-validation.

- Uses patient-cluster bootstrap sampling.



The split contained:



| Split | Patients |

|---|---:|

| Training | 254 |

| Testing | 45 |

| **Total** | **299** |



The selected Track B model is an ensemble of Random Forest and Logistic Regression.



The test interval-level ROC-AUC was 0.7266.



The patient-cluster bootstrap confidence interval was 0.5811–0.8460, based on 2,000 samples.



This is interval-level ROC-AUC, not a C-index. It should not be interpreted as a conventional survival concordance measure.



# 9. Model Artifact Verification



During testing, an important issue was identified in the saved Track A model bundle.



The `predict_a()` function requires the calibrator to transform model probabilities.



However, the training script originally saved the model without including the calibrator.



The relevant file was:



`scripts/train_cardiocare.py`



The bundle was corrected to include:



`"calibrator": result_a\["calibrator"]`



The corrected artifact was regenerated.



### Artifact verification output



The saved bundle was inspected and reported:



- `model`

- `calibrator`

- `features`

- `training_prior`

- `global_importance`

- `importance_mode`



The calibrator was confirmed to be present.



This resolved the mismatch between the training output and prediction-time requirements.



The artifacts were regenerated for both tracks:



- `models/agents/cardiocare/cardiocare_track_a.joblib`

- `models/agents/cardiocare/cardiocare_track_b.joblib`

- `models/agents/cardiocare/metadata.json`



# 10. CardioCare Agent Integration Review



The agent adapter was reviewed in:



`src/cad_hybrid/agents/cardiocare.py`



Two independent agents are implemented.



## 10.1 CardioCare CAD agent



Agent name:



`cardiocare_cad`



Target:



`cad`



The agent:



- Uses the Track A model.

- Requires all 18 Track A input features.

- Maps canonical system fields to the original model features.

- Loads the saved Track A artifact.

- Calls the prediction function.

- Validates the returned probability and training prior.

- Produces a standardized agent report.

- Includes model provenance and caveats.

- Provides feature-importance information.



The CAD agent is configured for testing but is not currently permitted to contribute to the fusion vote.



## 10.2 CardioCare heart failure agent



Agent name:



`cardiocare_hf`



Target:



`hf_survival`



The agent uses Track B.



It requires:



- Confirmed heart failure diagnosis.

- Age.

- Sex.

- Anaemia.

- Diabetes.

- High blood pressure.

- Smoking.

- Ejection fraction.

- Serum creatinine.

- Serum sodium.



The adapter checks the predicted survival outputs for finite values, valid probability ranges and non-increasing survival probabilities.



The output includes:



- Predicted days.

- Survival probability.

- 90-day risk.

- 300-day risk.



The agent does not return a CAD probability or training prior.



This separation prevents the heart failure survival output from being incorrectly treated as a CAD prediction.



# 11. Field Mapping and Configuration Verification



Two configuration files were examined:



- `configs/agents.yaml`

- `configs/fields.yaml`



## 11.1 Field mapping



The field configuration contains 28 canonical fields.



These include demographic, clinical, laboratory, ECG-related and heart failure-related fields.



The CardioCare CAD and HF feature mappings were reviewed against the adapter implementation.



The required mappings were confirmed to be available.



## 11.2 Agent configuration



The CardioCare entries were verified as follows:



| Agent | Target | Evidence group | Voting | Weight | Enabled |

|---|---|---|---|---:|---|

| cardiocare_cad | cad | clinical | No | 0.0 | Yes |

| cardiocare_hf | hf_survival | hf | No | 0.0 | Yes |



The configuration permits the agents to be enabled for integration testing without allowing them to contribute to the existing fusion score.



This is an important distinction between testing an agent and authorizing its output to influence a combined risk estimate.



## 11.3 Configuration validation



The YAML validation output confirmed:



- Five agents were detected.

- Twenty-eight fields were loaded.

- Both CardioCare agents were present.

- Required fields were available.

- CardioCare CAD target and evidence group were correct.

- CardioCare HF target and evidence group were correct.



No configuration change was identified as necessary during this review.



# 12. Automated Testing and Verification



The implementation was tested at two levels.



## 12.1 Targeted CardioCare tests



The targeted CardioCare tests were executed after regenerating the corrected model artifacts.



Result:



**12 tests passed.**



This confirmed the relevant CardioCare functionality, including the corrected prediction artifact.



## 12.2 Full test suite



The complete project test suite was executed.



Final result:



| Test result | Count |

|---|---:|

| Passed | 106 |

| Failed | 0 |

| Warnings | 3 |



Execution time was approximately 97.68 seconds.



The warnings were SHAP `PendingDeprecationWarning` messages and did not cause test failures.



**Overall automated test result: PASSED.**



# 13. Source Code Review and Git Change Assessment



The final source changes were reviewed before preparing the contribution.



The latest tracked change summary was:



| File | Change |

|---|---|

| `cardiocare/model.py` | Updated model calibration implementation |

| `scripts/train_cardiocare.py` | Added calibrator to saved model bundle |

| `models/agents/cardiocare/metadata.json` | Updated evaluation metrics and calibration metadata |



The command:



`git diff --check`



completed without reporting whitespace errors.



The source review examined:



- OOF calibration logic.

- Probability-to-logit conversion.

- Model refitting.

- Prediction-time calibration.

- Saved artifact structure.

- Updated evaluation metadata.



The source changes were considered consistent with the corrected training and prediction workflow.



Additional backup files were retained locally for comparison and were not intended to be part of the final contribution.



# 14. Evidence and Investigation Files



The following files support the Day 5 investigation.



| File | Purpose |

|---|---|

| `reports/split_manifest.csv` | Records split assignments |

| `reports/split_verification.txt` | Records split verification results |

| `reports/phash_20_pairs.csv` | Lists 20 pHash candidate pairs |

| `reports/phash_20_pairs.png` | Visual summary of candidate pairs |

| `reports/phash_distance2_comparison.png` | Comparison of five distance-2 pairs |

| `reports/final_normal_pair.png` | Detailed comparison of a Normal image pair |

| `models/agents/cardiocare/metadata.json` | Model evaluation and calibration metadata |



These files provide supporting evidence for the investigation and model evaluation.



# 15. Limitations and Unresolved Technical Questions



Although the investigation produced positive verification results, several questions remain.



## 15.1 Patient-level independence



The duplicate-group split verification operates on existing duplicate-group identifiers.



It does not establish that every ECG image belongs to a unique patient.



Patient identifiers or source-level patient records are required for stronger patient-level independence verification.



## 15.2 Exact duplicate-group provenance



The SHA-256 consistency checks passed for the existing duplicate-group file.



However, the exact process used to create `exact_duplicate_groups.csv` was not located.



Its original provenance should be documented if available.



## 15.3 pHash false positives and false negatives



pHash measures visual similarity.



It may identify images with similar layouts even when their waveform content differs. It may also miss duplicates if image formatting or preprocessing differs significantly.



A waveform-region-only comparison and SSIM-based investigation could provide additional evidence.



## 15.4 Temporal leakage



The measurement timestamps for EF-TTE and other clinical fields were unavailable.



It remains necessary to establish whether each measurement was collected before the intended prediction time.



## 15.5 External validation



Track A uses an Iranian angiography-referred cohort.



Track B uses a separate 299-patient heart failure dataset.



Neither result establishes general clinical performance in an Indian or other external population.



## 15.6 Calibration evaluation



The updated OOF Platt calibration produced a slightly improved Brier score and slightly lower test AUC.



Calibration should continue to be assessed on independent external data and should not be assumed to generalize automatically.



## 15.7 Fusion readiness



The CardioCare agents remain configured with voting disabled and weight zero.



Their outputs should not be added to the final fusion calculation until their target definitions, probability calibration, training priors, evidence groups and cross-agent dependencies have been reviewed.



# 16. Recommendations and Future Work



The following work is recommended before broader integration:



1\. Obtain patient-level identifiers, if available, to verify patient independence across splits.

2\. Document the provenance of the original duplicate-group file.

3\. Perform waveform-region-specific similarity analysis on suspicious pHash pairs.

4\. Establish measurement timestamps for EF-TTE and other clinical features.

5\. Conduct external validation using an independent cohort.

6\. Continue calibration evaluation on independent data.

7\. Confirm the standardized agent response contract with the `cad-agentic` repository.

8\. Keep CardioCare agents non-voting until the fusion design is formally reviewed.

9\. Preserve model metadata and dataset verification reports alongside the artifacts.

10\. Maintain reproducible training and testing procedures.



# 17. Day 5 Completion Summary



| Work item | Status |

|---|---|

| CardioCare feature review | Completed |

| Additional dataset column inspection | Completed |

| Single-feature AUROC analysis | Completed |

| Potential target leakage review | Completed |

| Temporal leakage limitation documented | Completed |

| Original ECG preprocessing review | Completed |

| Duplicate-group CSV investigation | Completed |

| SHA-256 consistency verification | Passed |

| Original group-aware split code review | Completed |

| Current split verification | Passed |

| Cross-split pHash analysis | Completed |

| Visual inspection of candidate pairs | Completed |

| pHash cleaning script review | Completed |

| CardioCare calibration code review | Completed |

| Saved model artifact verification | Passed |

| Agent adapter review | Completed |

| YAML configuration validation | Passed |

| Targeted tests | 12 passed |

| Full project tests | 106 passed |

| Git diff whitespace check | Passed |

| Final integration handover | To be prepared |



# 18. Overall Conclusion



Day 5 involved a detailed investigation of the CardioCare model inputs, ECG dataset preparation, duplicate-group assignments, split integrity, visual similarity candidates, model calibration, saved artifacts and agent configuration.



The selected CardioCare Track A feature set was reviewed against the original dataset. Several angiographic and outcome-proximal fields were identified and confirmed to be excluded from the selected model inputs.



The original ECG project was found to use group-aware stratified splitting. The existing duplicate-group assignments were preserved during integration, and the current verification confirmed zero duplicate-group overlap between training, validation and testing.



An additional pHash investigation identified 20 cross-split candidate pairs. Visual inspection did not confirm exact duplicates among the reviewed pairs. These findings should not be interpreted as proof of patient-level independence.



The CardioCare calibration implementation and saved model bundle were also reviewed and corrected. The calibrator is now included in the Track A artifact, and the targeted and full test suites passed.



The CardioCare agents and their field mappings were reviewed. They remain enabled for testing but excluded from fusion voting.



**Final assessment:** The CardioCare integration has passed its current automated tests and duplicate-group split verification. The model and adapter are ready for the next stage of technical integration review, subject to the documented limitations and external validation requirements.



This work represents an academic research and software-integration prototype. It does not establish clinical readiness, diagnostic validity or suitability for independent patient care.





