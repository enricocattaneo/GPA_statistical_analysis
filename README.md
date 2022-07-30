# College GPA: between Gender and IQ/Preparation

## Table of Contents

- [Problem Description](#problemdescription)
- [Data Description](#datadescription)
- [Regression Models](#regressionmodels)
- [Results](#results)


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
We start from two simple linear regressions and conclude our analysis with a multiple regression model. For each model, we also computed confidence intervals and conducted hypothesis testing.

##### SIMPLE LINEAR REGRESSION WITH GPA_FY ~ SAT_VERBAL

MODEL INFO:
Observations: 1000
Dependent Variable: gpa_fy
Type: OLS linear regression 

MODEL FIT:
F(1,998) = 191.67, p = 0.00
R² = 0.16
Adj. R² = 0.16 

Standard errors: OLS
-----------------------------------------------
                    Est.   S.E.   t val.      p
----------------- ------ ------ -------- ------
(Intercept)         0.70   0.13     5.41   0.00
sat_verbal          0.04   0.00    13.84   0.00
-----------------------------------------------

As expected, the coefficient for the sat_verbal explanatory variable has a positive sign (0.036). In this case, the regression's F-test confirms its significance (we reject the null hypothesis). The confidence intervals (obtained with 3 different methods: regression table, percentile method, and standard error method) for this coefficient do not contain the value 0, and the p-value is equal to zero. Thus, we are also inclined to reject the t-test's null hypothesis H0. The sat_verbal's coefficient is statistically significant and, as seen from the correlation value, as explanatory power for the response variable GPA_fy. 


#### SIMPLE LINEAR REGRESSION WITH GPA_FY ~ SEX
```{r}
data_model1 %>%  # visualization
  ggplot(aes(x = sex, y = gpa_fy)) +
  geom_boxplot(aes(fill = sex)) +
  labs(x = "Sex",
       y = "College GPA",
       title = "Relation between College GPA and Gender") +
  scale_fill_manual(values=c("#3399FF", "#FF6666"))

simple_model_sex = lm(gpa_fy ~ sex, data = data_model1) # fit regression model
get_regression_table(simple_model_sex) # get regression table
```

##### Confidence Intervals 

```{r}
bootstrap_diff_in_means = data_model1 %>%  # bootstrap diff in means
  specify(gpa_fy ~ sex) %>% 
  generate(reps = 10000, type = "bootstrap") %>% 
  calculate(stat = "diff in means", order = c("F", "M"))

percentile_ci_sx = bootstrap_diff_in_means %>% # percentile method 
  get_confidence_interval(type = "percentile", level = 0.95)
percentile_ci_sx

observed_diff_sx = data_model1 %>% # standard error method 
  specify(gpa_fy ~ sex) %>% 
  calculate(stat = "diff in means", order = c("F", "M"))

se_ci_sx = bootstrap_diff_in_means %>% 
  get_ci(level = 0.95, type = "se", point_estimate = observed_diff_sx)
se_ci_sx

visualize(bootstrap_diff_in_means)+ 
  shade_confidence_interval(endpoints = percentile_ci_sx, fill = NULL, 
                            linetype = "solid", color = "green") + 
  shade_confidence_interval(endpoints = se_ci_sx, fill = NULL, 
                            linetype = "dashed", color = "red") +
  shade_confidence_interval(endpoints = c(0.057, 0.24), fill = NULL, 
                            linetype = "dotted", color = "blue") +
  geom_vline(xintercept = 0)
```

##### Hypothesis Testing

```{r}
# HYPOTHESIS TESTING 
null_distribution_sx = data_model1 %>% # null distribution
  specify(gpa_fy ~ sex) %>% 
  hypothesize(null = "independence") %>% 
  generate(reps = 10000, type = "permute") %>% 
  calculate(stat = "diff in means", order = c("F", "M"))

visualize(null_distribution_sx) + # visualize the p-value
  shade_p_value(obs_stat = observed_diff_sx, direction = "both")

null_distribution_sx %>% # what percentage of the null distr. is shaded?
  get_p_value(obs_stat = observed_diff_sx, direction = "both")

anova(simple_model_sex)
```

The coefficient of sexF represents the offset of the group Female from the reference group Male. The difference is positive, so it means that the Female group has a higher GPA score than the Male group. Using the ANOVA table we can see that the regression is significant at the 0.001 level (F-test statistic).  The confidence intervals (obtained with 3 different methods) do not contain the value 0, and the coefficients' p-value is close to zero and smaller than every possible significance level. Thus, we are inclined to reject the null hypothesis H0, we can say that there is a positive and statistically significant difference between the two groups.


After seeing that the variable sat_verbal and sex are both significative, when taken separately in a simple linear model, and they both have some explanatory power over gpa_fy. We now try to enrich our analysis by constructing a multiple linear regression model. As said before it is preferable not to include the variables sat_total and sat_math because they are both correlated with the other explanatory variable sex. We could run into some multicollinearity problems, and to avoid that we have chosen the variable sat_verbal because it is not correlated with gender. The other variable available is gpa_hs but it suffers from the same problem due to its link with the SATs variables, and it is personally not an additional source of interest in regards to our problem's analysis. 
For those reasons, we opt for a multiple regression model with two explanatory variables: a categorical one and a numerical one. We will understand if the addition of both those variables (that were significant when individually taken) will be relevant when taken into consideration together. For this, we will analyze the t-test and the F-test (both overall and partial). Comparing the F-test from the simple regression model and the multiple regression one, we will see if the addition of the variable is beneficial or not. 
We will consider both a parallel slopes model and an interaction model. The former is a model where the factor has an impact only on the mean of the response variable, while the latter is a model where the factor can also have an impact on the variation of the change in the mean of the response.  

#### MULTIPLE LINEAR REGRESSION WITH GPA_FY ~ SAT_TOTAL AND SEX
##### INTERACTION MODEL
```{r}
gpa_interaction = lm(gpa_fy ~ sat_verbal * sex, # fit regression model
                      data = data_model1, x = T)
get_regression_table(gpa_interaction) # get regression table 
summary(gpa_interaction)
anova(gpa_interaction)
```

The interaction model does not seem to be a good model for the data. From the t-test, we can understand that both the sexF coefficient and the interaction coefficient are not statistically significant, while we can reject the null hypothesis only for the sat_verbal coefficient. So the F group does not have a significant impact on the mean of the response variable with reference to the reference group (M). Also, the group does not have a significant impact on the variation of the response variable. The F-test seems to confirm this suggestion because we cannot reject the null hypothesis for the interaction term, so the change in the response in all groups can be considered equal. 

We try implementing a parallel slope model (without an interaction term) to solve this problem. 

##### PARALLEL SLOPE (DUMMY) MODEL
```{r}
gpa_parallel_slopes = lm(gpa_fy ~ sat_verbal + sex, # fit regression model
                         data = data_model1, x = T)
get_regression_table(gpa_parallel_slopes) # get regression table
summary(gpa_parallel_slopes)
anova(gpa_parallel_slopes)
```

The parallel slope model performs way better than the interaction model.
The t-test, bring us to reject the null hypothesis for the coefficients of both variables at any level of significance. From the F-test, we can understand that this multiple regression model performs better than the simple regression model with the only sat_verbal as explanatory variable (using both overall and partial F test). Both the sat_verbal score and the gender of the student are significant to explain the GPA for the first college year.

##### MOST SUITABLE REGRESSION MODEL VIA BACKWARD MODEL SELECTION 
```{r}
gpatry = lm(gpa_fy ~ ., # fit regression model
          data = sat_gpa, x = T)

gpa_back <- step(gpatry, direction = "backward")

get_regression_table(gpa_back) # get regression table
summary(gpa_back)
anova(gpa_back)
```

In this last case, we have considered all the variables in our dataset. Via the backward model selection, we find the most suitable regression model. With this method, at each step, we exclude the least statistically significant variable. At the end of the process, we obtain a model with four statistically significant explanatory variables (sex, sat_verbal, sat_math, and gpa_hs). 
By construction, the t-test brings us to reject the null hypothesis for the coefficients of all four variables at any level of significance. From the F-test, we can understand that this multiple regression model performs better than both the simple regression models and the parallel slopes regression model (with two explanatory variables). So, all the variables are significant to explain the GPA for the first college year.

#### DIAGNOSTIC PLOTS FOR LINEAR REGRESSION ANALYSIS

We would focus on some diagnostic plots for linear regression analysis of our last two models: the parallel slope model with two explanatory variables and the model found via backward selection with four explanatory variables. We can check if a model works well for data in many different ways. We already have paid great attention to regression results, such as coefficients, p-values (and later R-squared) that tell us how well a model represents given data. However, also residuals could show how poorly a model reproduces data. Residuals are leftover of the outcome variable after fitting a model (predictors) to data, and they could reveal unexplained patterns in the data by the fitted model. 

##### RESIDUALS 

Parallel slope model

```{r}
res1 = residuals(gpa_parallel_slopes) 
hist(res1, col = "green")
shapiro.test(res1)
qplot(sample = .stdresid, data = gpa_parallel_slopes) +
  geom_abline(color = "green")
skewness(res1)
kurtosis(res1)
```

Backward model

```{r}
res2 = residuals(gpa_back) 
hist(res2, col = "steelblue")
shapiro.test(res2)
qplot(sample = .stdresid, data = gpa_back) +
  geom_abline(color = "steelblue")
skewness(res2)
kurtosis(res2)
```

For both regression models, we have plotted a histogram of the residuals, the normal QQ plot and computed the Shapiro-Wilk test. All these computations/visualizations are used to verify the normality of the residuals. The histogram gives us a first and easily understandable visualization of the distribution of the residuals. The normal QQ plot shows if residuals follow a straight line well or if they deviate severely. It’s good if residuals are lined well on the straight dashed line. In the Shapiro-Wilk test, the null hypothesis is that the population is normally distributed. Thus, if the p-value is less than the chosen alpha-level, then the null hypothesis is rejected and there is evidence that the data tested are not normally distributed. On the other hand, if the p-value is greater than the chosen alpha level, then the null hypothesis (that the data came from a normally distributed population) can not be rejected.
In our particular case for both models' residuals, we observe that the visualizations suggest normally distributed residuals while the Shapiro-Wilk test suggests the opposite (in both models we reject the null hypothesis of normal distribution). This discordance of results might be due to the sample size or the fact that we are working with multiple regression models (the Shapiro-Wilk test performs better with small sample size and a single independent variable). To have a better understanding, we have also calculated the skewness and kurtosis of the residuals, which in both cases suggest normal distribution.

##### RESIDUALS VS FITTED
This plot shows if residuals have non-linear patterns. There could be a non-linear relationship between predictor variables and an outcome variable, and the pattern could show up in this plot if the model doesn’t capture the non-linear relationship.

Parallel slope model

```{r}
plot(res1, ylab = "Residuals")
plot(gpa_parallel_slopes$fitted.values, res1, xlab = "GPA", ylab = "Residuals")
abline(h=0)
```

Backward model

```{r}
plot(res2, ylab = "Residuals")
plot(gpa_back$fitted.values, res2, ylab = "Residuals", xlab = "GPA")
abline(h=0)
```

For both models, we have plotted residuals vs fitted values. Looking at all the plots we can observe that the residuals definitely show randomness. We can use this plot to understand if residuals have non-linear patterns. Finding equally spread residuals around a horizontal line without distinct patterns, that is a good indication you don’t have non-linear relationships. Furthermore, the residuals do not appear to be equally variable across the entire range of fitted values. So, we can conclude that there is no discernible non-linear trend to the residuals and there is no indication of non-constant variance.


##### R-SQUARED
```{r}
summary(simple_model_sat)$r.squared 
summary(simple_model_sex)$r.squared 
summary(gpa_parallel_slopes)$r.squared 
summary(gpa_back)$r.squared 
```

R-squared is a goodness-of-fit measure for linear regression models. This statistic indicates the percentage of the variance in the dependent variable that the independent variables explain collectively. R-squared measures the strength of the relationship between your model and the dependent variable on a 0 to 100 scale. 
In our particular case, we have computed the R-squared for our two simple regression models, and for two multiple regression models. The R-squared for the simple regression model with sat_verbal as an explanatory variable is 16% definitely not a high value but we can see that this variable has some clear explanatory power on the data. Instead, the R-squared for the simple regression model with sex as an explanatory variable is only 1% suggesting an extremely weak relationship between this model and the dependent variable. The parallel slope model with sat_verbal and sex as independent variables has a very similar R-squared (17%) as the simple regression model with sat_verbal, highlighting the relative strength of the sat_verbal variable on the sex variable. The last model obtained with backward selection and with four independent variables has the highest R-squared value at 32%. It is still not a symptom of a strong relationship between the model and the dependent variable (gpa_fy) but is the best we got.


### FINAL COMMENT

Our analysis aimed to better understand the relationships that link college GPA with gender (a categorical variable) and the results from the SAT (a numerical variable). In the Explanatory Data Analysis, we looked at the raw values, provided summary statistics for the variables of the dataset, and some visualizations. Starting with some simple univariate data visualizations and ending with more complex multivariate data visualizations we gain a better understanding of the variables and the overall problem. Then we started with two simple linear regression models: both with the college GPA as the response variable, the first with the verbal SAT as an explanatory variable while the second with gender as the independent variable. In both cases, we found a statistically significant coefficient for our regressor. The coefficient for the sat_verbal explanatory variable had a positive sign of 0.036, as we expected. While the coefficient of the sex-F variable (representing the offset between the group Male and the group Female) had a value of 0.149. After seeing that the variables sat_verbal and sex are both significative when taken separately in a simple linear model, we enriched our analysis by constructing a multiple linear regression model. Using both gender and SAT verbal as explanatory variables of our multiple regression model we had two possible approaches to choose from: an interaction model and a parallel slopes model. We started analyzing the interaction model that turned out to be not good for our data, the interaction term was not statistically significant. For this reason, we opted for a parallel slope model that differs from the interaction model because it does not contain any interaction term. The parallel slope model performs way better than the interaction model. Both explanatory variables were statistically significant so not only when taken separately but also when taken together both sat_verbal score and gender are significant to explain the GPA for the first college year. This was the model that we thought about as a response to the problem that we posed at the beginning of our analysis. To produce a further and more complete analysis we decided to consider all the variables in our dataset. We tried to find the most suitable regression model via backward model selection. With this method, at each step, we exclude the least statistically significant variable. At the end of the process, we obtained a model with four statistically significant explanatory variables (sex, sat_verbal, sat_math, and gpa_hs). From the F-test, we understood that this multiple regression model performs better than both the simple regression models and the parallel slopes regression model (with two explanatory variables). At the end of our analysis, we focused on some diagnostic plots and tests for our two best models: the parallel slopes model (with two independent variables) and the model produced via backward selection (with 4 independent variables). We first observed the residuals of our models producing a histogram, the Shapiro-Wilk normality test, the normal QQ plot, and skewness and kurtosis. All these components of this evaluation brought us to have a better understanding of the normality or not of the residuals' distribution. We obtained some discordant results (all the visualizations and the kurtosis and skewness indexes suggest normality, while the Shapiro-Wilk test suggests not normal distribution). This difference in the results may be due to due to the sample size of our data or the fact that we are working with multiple regression models (the Shapiro-Wilk test performs better with a small sample size and a single independent variable). Then we plotted the residuals and the fitted values, the residuals of both models showed randomness and they are not characterized by any non-linear pattern. So, we concluded that no discernible non-linear trend was noticeable. Lastly, we computed the R-Squared (goodness of fit measure for linear regression models) for each of the four models that we had previously considered. For all four models, the R-squared assumed low values. In particular, the simple regression model with the only gender as an independent variable had an R-squared of only 1%. The model with the best performance in terms of R-squared was the last model with four independent variables (32%).

To find the best model we have to make a choice. On the one hand, we could opt for the model that better explains our data, in that case, we would choose the multiple regression with four explanatory variables. On the other hand, we could choose the parallel slope model with two explanatory variables that, differently from the previously consider model, is not characterized by multi-collinearity problems between regressors but that has a worse fit to the data (R-squared is equal to 17%). So the choice would be between a model with better performance but less statistically sound, and a model with lower performance but more sound. However, both of those models do not have good performances in absolute values (the introduction of polynomial variables does not bring any gain in performance).

The evaluations of the two models are based on two concepts: bias and performance. With our analysis of residuals, we found out that both models are unbiased, meaning that the fitted values are not systematically too high or too low anywhere in the observation space. Even though we obtained low R-squared values that do not mean that the models we constructed have no relevance. As said in the literature some fields of study have an inherently greater amount of unexplainable variation. In these areas, your R2 values are bound to be lower. For example, studies that try to explain human behavior generally have R2 values less than 50%. Fortunately, even with a low R-squared value but statistically significant independent variables, we can still draw important conclusions about the relationships between the variables. Statistically significant coefficients continue to represent the mean change in the dependent variable given a one-unit shift in the independent variable. In fact from the parallel slope model, we found out that both the verbal SAT and the gender have relationships with the college GPA, even if weakly linked (especially the gender). From the four explanatory variables model, we understand that also the math SAT and the high school GPA (even if characterized by multicollinearity) have explanatory power over the college GPA. Clearly, being able to draw conclusions like this is important.
