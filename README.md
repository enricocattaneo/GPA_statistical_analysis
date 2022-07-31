# College GPA: between Gender and IQ/Preparation

## Table of Contents

- [Problem Description](#problemdescription)
- [Data Description](#datadescription)
- [Regression Models](#regressionmodels)
- [Diagnostic Analysis](#diagnostics)
- [Conclusion](#conclusion)


<a id='problemdescription'></a>

## Problem Description
##### Does gender influence College GPA? Is the SAT score a valuable indication of future GPA?
Our analysis will investigate the differences in GPA between genders (binary) and the relationship between college GPA and SAT scores (taken during High School). Considering the SAT as an approximation of the IQ (or, in a more precise way, of past preparation), we can expect a positive relationship between this variable and the GPA. While the relationship between GPA and gender is of less intuitive interpretation. We will study our sample data (considered as randomly selected and representative) to understand better these relationships' signs and values between those variables. 

<a id='datadescription'></a>
## Data Description

Each observation (row) in the data frame represents a student. The dataset contains 1000 observations (students) or rows and 7 variables or columns (including the id column). 

* X1: the first column is automatically given a name during the import phase because it does not have one. It is a sequence to identify each observation. We will rename it later as id.
* sex: (binary) gender of the student. The second variable is categorical, categorized as "Female" and "Male". We will transform it later into a factor, and we will give it "M" and "F" values for simplicity.
* sat_verbal: Verbal SAT percentile. It is a numerical variable that can go from 0 to 100. In our dataset, the maximum value is 76, and the minimum is 24.
* sat_math: Math SAT percentile. It is a numerical variable that can go from 0 to 100. In our dataset, the maximum value is 77 and, the minimum one is 29.
* sat_total: Total of verbal and math SAT percentiles. It is a numerical variable that can go from 0 to 200. In our dataset, the maximum value is 144, and the minimum one is 53. 
* gpa_hs: High school grade point average. It is a categorical variable, categorized as "high" or "low". For simplicity, we will change it in a factor and change the categories in "H" or "L".
* gpa_fy: First year (college) grade point average. It is a numerical variable that can go from 0 to 4, and it has decimals. In our dataset, the maximum value is 4, and the minimum one is 0.

##### Response and Explanatory Variables 
* Response: gpa_py (numerical variable).
* Explanatory: sat_verbal (numerical variable), sex (categorical variable).
*In our analysis, we will use sat_verbal instead of sat_total or sat_math to avoid multicollinearity problems in the multiple linear regression (differently from the SAT total score and math score, there is no significant difference between genders for the SAT verbal score).*

## Exploratory Data Analysis (EDA)
Summary statistics of the variables. Both univariate and multivariate data visualizations. 

<a id='regressionmodels'></a>
## Regression Models
We start from two simple linear regressions and conclude our analysis with a multiple regression model. For each model, we also computed manually both confidence intervals and hypothesis testing.



##### Main Regression Models Results
<img width="1334" alt="Schermata 2022-07-30 alle 18 46 55" src="https://user-images.githubusercontent.com/80990030/181933561-f0093165-e1b1-487f-904d-fdb458d7596e.png">

* **Simple Linear model with gpa_fy ~ sat_verbal.**
* **Simple Linear model with gpa_fy ~ sex.**
* **Multiple linear regression with gpa_fy ~ sat_total and sex: *Interaction model.***
* **Multiple linear regression with gpa_fy ~ sat_total and sex: *Parallel Slope model.***
* **Multiple linear regression: *Most Suitable Model Via Backward Model Selection.***

<a id='diagnostics'></a>
## Diagnostic Analysis
We would focus on some diagnostic plots for linear regression analysis of our last two models: the parallel slope model with two explanatory variables and the model found via backward selection with four explanatory variables (two best performances). 

* Residuals Analysis
* Residuals vs Fitted Values
* R-Squared


## Conclusion

Our analysis aimed to better understand the relationships that link college GPA with gender (a categorical variable) and the results from the SAT (a numerical variable).

To find the best model we have to make a choice. On the one hand, we could opt for the model that better explains our data, in that case, we would choose the multiple regression with four explanatory variables. On the other hand, we could choose the parallel slope model with two explanatory variables that, differently from the previously consider model, is not characterized by multi-collinearity problems between regressors but that has a worse fit to the data (R-squared is equal to 17%). So the choice would be between a model with better performance but less statistically sound, and a model with lower performance but more sound. However, both of those models do not have good performances in absolute values (the introduction of polynomial variables does not bring any gain in performance).

The evaluations of the two models are based on two concepts: bias and performance. With our analysis of residuals, we found out that both models are unbiased, meaning that the fitted values are not systematically too high or too low anywhere in the observation space. Even though we obtained low R-squared values that do not mean that the models we constructed have no relevance. As said in the literature some fields of study have an inherently greater amount of unexplainable variation. In these areas, your R2 values are bound to be lower. For example, studies that try to explain human behavior generally have R2 values less than 50%. Fortunately, even with a low R-squared value but statistically significant independent variables, we can still draw important conclusions about the relationships between the variables. Statistically significant coefficients continue to represent the mean change in the dependent variable given a one-unit shift in the independent variable. In fact from the parallel slope model, we found out that both the verbal SAT and the gender have relationships with the college GPA, even if weakly linked (especially the gender). From the four explanatory variables model, we understand that also the math SAT and the high school GPA (even if characterized by multicollinearity) have explanatory power over the college GPA. Clearly, being able to draw conclusions like this is important.
