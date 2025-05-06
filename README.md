# SNAP-Gun-Violence
[Click here](https://nbviewer.org/github/hmcooper1/SNAP-Gun-Violence-Predictor/blob/main/snap_gun_violence.ipynb) to view the full notebook with choropleth maps on NBViewer.
## Research Question
To what extent is food security, a basic need, through SNAP associated with mass shootings gun violence in the United States?
## Overview
This project explores the relationship between access to basic food benefits, provided by the Supplemental Nutrition Assistance Program (SNAP), and gun violence rates. We hypothesize that food security fosters societal stability and reduces gun violence. Using data from 2017 to 2019, we analyze mass shootings at the county level, incorporating data from the U.S. Census and SNAP participation records.
## Data Sources
- [Gun Violence Data](https://www.gunviolencearchive.org/reports/mass-shooting?year=2023): Mass shooting data (2017-2019) retrieved from  the Gun Violence Archive
- [City and Census Data](https://data.census.gov/table/ACSST1Y2016.S1701?q=percent%20of%20population%20below%20the%20poverty%20line%20by%20county&g=040XX00US06): US Census Burea's data on poverty status and other demographic information by county
- [SNAP Data](https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap): Statistics on the number of individuals receiving SNAP benefits
## Models
- Regression models (predicting the proportion of mass shooting victims by county): Ordinary Least Squares, Decision Tree Regression, AdaBoost Regression
- Classification models (predicting high or low gun violence by county): Logistic Regression, Decision Tree Classifier, AdaBoost Classifier
## Key Findings
Although SNAP participation was previously hypothesized to be a key driver of mass shooting incidence, it was not as strong a predictor as other demographic features with classification models. Nonetheless, variables related to SNAP participation still showed a positive association with gun violence rates — areas with higher SNAP participation tended to experience higher rates of mass shootings. However, these factors were ultimately less predictive than demographic variables. Education levels and race were the most predictive features for mass shooting rates. Classification models outperformed regression models, suggesting that mass shooting gun violence is better categorized than predicted as a continuous variable. The Adaboost Classifier performed best, achieving: 85% accuracy, 76% AUC-ROC score, and 80% precision. Further policy discussions could explore the effects of educational funding and social programs on reducing gun violence.
## Notes
This project was completed by Helena Wu, Jane Lee, and Hannah Cooper during Jonathan Marhsall's Data, Prediction & Law course (LEGALST123) at UC Berkeley.
