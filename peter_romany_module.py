import scipy
import re
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from scikit_posthocs import posthoc_dunn
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def dealing_with_outlier(df, variable, show_outliers=True):
    quartiles = df[variable].quantile([0.25, 0.50, 0.75])
    m = quartiles[0.25] - (1.5 * (quartiles[0.75] - quartiles[0.25]))
    p = quartiles[0.75] + (1.5 * (quartiles[0.75] - quartiles[0.25]))
    indices_of_outlier = list(df[~((df[variable] >= m) & (df[variable] <= p))].index)
    indices_of_outlier_p = list(df[((df[variable] > p))].index)
    indices_of_outlier_m = list(df[((df[variable] < m))].index)
    values_of_outlier = list(df[variable][df[~((df[variable] >= m) & (df[variable] <= p))].index])
    indices_and_values_of_outlier = dict(zip(indices_of_outlier, values_of_outlier))
    # print(f'indices_and_values_of_outlier_of {variable}',indices_and_values_of_outlier)
    if show_outliers:
        print(f'm= {m}',f'p= {p}',)
        print(f'values_of_outlier_p_of {variable}',list(df[variable][df[((df[variable] > p))].index]))
        print(f'values_of_outlier_m_of {variable}',list(df[variable][df[((df[variable] < m))].index]))
    # print(f'indices_of_outlier_of {variable}',indices_of_outlier)
    # print(indices_of_outlier_m,indices_of_outlier_p)
    df.loc[indices_of_outlier_p, [variable]] = p
    df.loc[indices_of_outlier_m, [variable]] = m
    # return m,p,indices_of_outlier,indices_of_outlier_m,indices_of_outlier_p,indices_and_values_of_outlier
    return indices_and_values_of_outlier


def insights_by_descriptive_analytics(df,variable):
    print('perform insights_by_descriptive_analytics')
    mode=df[variable].mode()
    median=df[variable].median()
    std=df[variable].std()
    mean=df[variable].mean()
    boundaryPlus=mean+std
    boundaryMinus=mean-std
    CV=((std/mean)*100)
    CV_Rate=(((df[variable][df[((df[variable]>=boundaryMinus)&(df[variable]<=boundaryPlus))].index].count())/df[variable].count())*100)
    print('===============================',variable,'===========================================')
    print('minus=',boundaryMinus,'mean=',mean,'plus=',boundaryPlus,'\n','CV=',CV,'%','CV_Rate=',CV_Rate,'%')
    print('=================================================================================')
    # print((((df[variable][df[((df[variable]>=boundaryMinus)&(df[variable]<=boundaryPlus))].index]))).sort_values())
    # print((((df[variable][df[~((df[variable]>=boundaryMinus)&(df[variable]<=boundaryPlus))].index]))).sort_values())
    skewness=df[variable].skew()
    if(skewness>0):
        location='most in low [where mode locate (mode < median= {1} < mean= {2})]'.format(mode,median,mean)
        NO_location=df[variable][df[((df[variable]<=mean))].index].count()
    elif(skewness<0):
        location='most in high [where mode locate (mean={0} < median= {1} < mode)]'.format(mean,median,mode)
        NO_location=df[variable][df[((df[variable]>=mean))].index].count()
    else:
        location='most in mean [where mode locate (mean={0} = median= {1} = mode)]'.format(mean,median,mode)
        NO_location=df[variable][df[((df[variable]>=boundaryMinus)&(df[variable]<=boundaryPlus))].index].count()
    N=df[variable].count()
    print('skewness=',skewness,'rate',(NO_location/N)*100,'%','[',NO_location,"out of",N,"]",'[',location,']')
    print('==============================================================================================================================================================\n')


'''===============  Inferential Statistics  ==============='''

def check_normality(var1,var2):
    # perform Shapiro-Wilk test for normality if N<2000
    def check_normality_shapiro(data):
        print('===============================', data.name, '==================================')
        test_stat_normality, p_value_normality = scipy.stats.shapiro(data)
        print("p_value_normality:", p_value_normality)
        if p_value_normality < 0.05:
            print("Reject null hypothesis >> The data is not normally distributed")
        else:
            print("Fail to reject null hypothesis >> The data is normally distributed")
        print('========================================================================')

    # perform Kolmogorov-Smirnov test for normality if N>2000
    def check_normality_kolmogorov(data):
        print('===============================', data.name, '==================================')
        test_stat_normality, p_value_normality = scipy.stats.kstest(data, 'norm')
        print("p_value_normality:", p_value_normality)
        if p_value_normality < 0.05:
            print("Reject null hypothesis >> The data is not normally distributed")
        else:
            print("Fail to reject null hypothesis >> The data is normally distributed")
        print('========================================================================')

    if((len(var1)<=2000) & (len(var2)<=2000)):
        print('perform Shapiro-Wilk test for normality when N1 and N2 <=2000')
        check_normality_shapiro(var1)
        check_normality_shapiro(var2)
    elif((len(var1)>2000) & (len(var2)>2000)):
        print('perform Kolmogorov-Smirnov test for normality when N1 and N2 >2000')
        check_normality_kolmogorov(var1)
        check_normality_kolmogorov(var2)
    else: print('Please peter look at the samples size')


def check_variance_homogeneity(var1, var2):
    print('\nperform Levene test for homogeneity')
    print('===============================', '[', var1.name, '] with [', var2.name, ']',
          '==================================')
    test_stat_var, p_value_var= scipy.stats.levene(var1,var2)
    print("p_value_var:",p_value_var)
    if p_value_var <0.05:
        print("Reject null hypothesis >> The variances of the samples are different. Non Homogeneous.use only nonparametric")
    else:
        print("Fail to reject null hypothesis >> The variances of the samples are same. Homogeneous.use parametric or nonparametric")
    print('========================================================================')


#I will compare more than 2 means by my predefiend function
def independent_sample_ttest(var1,var2,equal_variance=True):
    print('\nperform independent_sample_ttest')
    print('\n===============================','[',var1.name,'] with [',var2.name,']', '==================================')
    ttest, p_value_ttest = scipy.stats.ttest_ind(var1, var2,equal_var=equal_variance)
    print("p_value_ttest:", p_value_ttest)
    if (p_value_ttest < 0.05):
        if (p_value_ttest < 0.01):
            print('The difference is significant with confidence level = 99% [ {0}`s mean = {1} |while| {2}`s mean = {3} ]'.format(
                var1.name, var1.mean(), var2.name, var2.mean()))
        else:
            print('The difference is significant with confidence level = 95% [ {0}`s mean = {1} |while| {2}`s mean = {3} ]'.format(
                var1.name, var1.mean(), var2.name, var2.mean()))
    else:
        print('The difference is [ not ] significant [ {0}`s mean = {1} |while| {2}`s mean = {3} ]'.format(
                var1.name, var1.mean(), var2.name, var2.mean()))
    print('========================================================================\n')


#I will compare more than 2 means by my predefiend function
def paired_sample_ttest(var1,var2):
    print('\nperform paired_sample_ttest')
    print('\n===============================','[',var1.name,'] with [',var2.name,']', '==================================')
    ttest, p_value_ttest = scipy.stats.ttest_rel(var1, var2)
    print("p_value_ttest:", p_value_ttest)
    if (p_value_ttest < 0.05):
        if (p_value_ttest < 0.01):
            print('The difference is significant with confidence level = 99% [ {0}`s mean = {1} |while| {2}`s mean = {3} ]'.format(
                var1.name, var1.mean(), var2.name, var2.mean()))
        else:
            print('The difference is significant with confidence level = 95% [ {0}`s mean = {1} |while| {2}`s mean = {3} ]'.format(
                var1.name, var1.mean(), var2.name, var2.mean()))
    else:
        print('The difference is [ not ] significant [ {0}`s mean = {1} |while| {2}`s mean = {3} ]'.format(
                var1.name, var1.mean(), var2.name, var2.mean()))
    print('========================================================================\n')


# i will do multiple regression  also by this function
def pearsonr(independent_variable,dependent_variable):
    print('\nperform pearsonr')
    print('\n===============================','[',independent_variable.name,'] with [',dependent_variable.name,']', '==================================')
    slope, intercept, rvalue, p_value_linregress, stderr = scipy.stats.linregress(independent_variable,dependent_variable)
    pearsonr, p_value_pearsonr = scipy.stats.pearsonr(independent_variable, dependent_variable)
    print("pearsonr:",pearsonr,'| r_interpretation:',r_interpretation(pearsonr),"| R-squared:", pearsonr ** 2, "%", "| p_value_pearsonr:",p_value_pearsonr,"| slope:", slope)
    if (p_value_pearsonr < 0.05):
        if (p_value_pearsonr < 0.01):
            print('there is a statistically significant linear association. with confidence level = 99%')
        else:
            print('there is a statistically significant linear association. with confidence level = 95%')
    else:
        print('there is [ not ] a statistically significant')
    print('========================================================================\n')


def spearmanr(independent_variable,dependent_variable):
    print('\nperform spearmanr')
    print('\n===============================','[',independent_variable.name,'] with [',dependent_variable.name,']', '==================================')
    spearmanr, p_value_spearmanr = scipy.stats.spearmanr(independent_variable, dependent_variable)
    print("spearmanr:",spearmanr,"| p_value_spearmanr:",p_value_spearmanr,'| r_interpretation:',r_interpretation(spearmanr),"| R-squared:", spearmanr ** 2, "%")
    if (p_value_spearmanr < 0.05):
        if (p_value_spearmanr < 0.01):
            print('there is a statistically significant linear association. with confidence level = 99%')
        else:
            print('there is a statistically significant linear association. with confidence level = 95%')
    else:
        print('there is [ not ] a statistically significant')
    print('========================================================================\n')


def r_interpretation(r):
    dependence_interpretation='Nothing'
    r=abs(r)
    if((r>=0.00) & (r<=0.09)):
        dependence_interpretation='Trivial or none'
    elif((r>=0.10) & (r<=0.29)):
        dependence_interpretation='Low to medium'
    elif((r>=0.30) & (r<=0.49)):
        dependence_interpretation='Medium to essential'
    elif((r>=0.50) & (r<=0.69)):
        dependence_interpretation='Essential to very strong'
    elif((r>=0.70) & (r<=0.89)):
        dependence_interpretation='Very strong'
    elif((r>=0.90) & (r<=1)):
        dependence_interpretation='Almost perfect'
    return dependence_interpretation


def sorted_zscore(var1,show_z_score=True):
    if show_z_score:
        print(scipy.stats.zscore(var1.sort_values()))# z-score and rank same value order but zscore can tell u where the mean is and other values from mean


def sorted_rank(variable,show_percent_rank=True):
    percent_rank = variable.rank(pct=True).sort_values()  # 2my position relative to other values #ascending=False |rank as first,second,third place and so on.
    if show_percent_rank:
        print(percent_rank)


def regex_patterns(df,variable,pattern,convert_to_str=False,return_match_indices=False,printing_result=True):
    tempo = df[variable]
    if convert_to_str:
        tempo = df[variable].astype('str')
    if printing_result:
        print(tempo[tempo.str.contains(re.compile(pattern))])
    if return_match_indices:
        match_indices = tempo[tempo.str.contains(re.compile(pattern))].index
        return match_indices


def regex_for_str(df,variable):

    print(df[variable][df[variable].str.contains(re.compile('[^a-zA-Z]'))])


def regex_for_float(df,variable):
    tempo=df[variable].astype('str')
    print(tempo[tempo.str.contains(re.compile('[^0-9.-]'))])
    print(tempo[tempo.str.contains(re.compile('[.]{2,}'))])


def regex_for_int(df,variable):
    tempo=df[variable].astype('str')
    print(tempo[tempo.str.contains(re.compile('[^0-9-]'))])


'''===============  Predictive Analytics  ==============='''


def linear_regression(df,independent_variable,dependent_variable,show_best_fit_line=True):
    print('\nperform linear_regression')
    print('\n==========================','Linear Regression [',independent_variable,'] with [',dependent_variable,']', '==============================')
    # Extracting X and y from the DataFrame
    X = df[independent_variable]
    y = df[dependent_variable]
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Creating and fitting the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Calculate R-squared using the score method
    r_squared = model.score(X, y)
    print(f"Coefficient of determination (R-squared): By {r_squared:.2f}%, my X(IV) explains or determines the change observed in y(DV).")
    # Calculate correlation coefficient from R-squared
    correlation = np.sqrt(r_squared)
    print(f"Pearson correlation coefficient (r): By {correlation:.2f}%, my points follow a constant rate of change")
    # Get the intercept from the fitted model
    intercept = model.intercept_
    print(f"Intercept(bias): An intercept of {intercept} might represent an estimated baseline.")
    # Get the slope (coefficient) from the fitted model
    slope = model.coef_[0]
    print(f"Slope (Coefficient): {slope}")

    # Making predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('\n========================Making predictions===========================')
    print("---------------------Making predictions done-----------------------")
    print(f"Evaluating the model prediction with Mean Squared Error: {mse:.2f}")
    print(f"R-squared of the prediction: {r2:.2f}")
    print('=====================================================================')
    if show_best_fit_line:
        # Plotting the fitted line
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual data')
        plt.plot(X_test, y_pred, color='red', label='Fitted line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
    print('========================================================================================================================\n')
    return model


def ridge_regression(df,independent_variable,dependent_variable,lamda,show_best_fit_line=True):
    print('\nperform ridge_regression')
    print('\n==========================','Ridge Regression [',independent_variable,'] with [',dependent_variable,']', '==============================')
    # Extracting X and y from the DataFrame
    X = df[independent_variable]
    y = df[dependent_variable]
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Creating and fitting the Ridge Regression model
    model = Ridge(alpha=lamda)
    model.fit(X_train, y_train)
    # Calculate R-squared using the score method
    r_squared = model.score(X, y)
    print(f"Coefficient of determination (R-squared): By {r_squared:.2f}%, my X(IV) explains or determines the change observed in y(DV).")
    # Calculate correlation coefficient from R-squared
    correlation = np.sqrt(r_squared)
    print(f"Pearson correlation coefficient (r): By {correlation:.2f}%, my points follow a constant rate of change")
    # Get the intercept from the fitted model
    intercept = model.intercept_
    print(f"Intercept(bias): An intercept of {intercept} might represent an estimated baseline.")
    # Get the slope (coefficient) from the fitted model
    slope = model.coef_[0]
    print(f"Slope (Coefficient): {slope}")

    # Making predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('\n========================Making predictions===========================')
    print("---------------------Making predictions done-----------------------")
    print(f"Evaluating the model prediction with Mean Squared Error: {mse:.2f}")
    print(f"R-squared of the prediction: {r2:.2f}")
    print('=====================================================================')
    if show_best_fit_line:
        # Plotting the fitted line
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual data')
        plt.plot(X_test, y_pred, color='red', label='Fitted line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Ridge Regression')
        plt.legend()
        plt.show()
    print('========================================================================================================================\n')
    return model


def logistic_regression(df,independent_variable,dependent_variable,max_iters=100,regularization=1.0,show_n_iter=True):
    print('\nperform logistic_regression')
    print('\n==========================','Logistic Regression [',independent_variable,'] with [',dependent_variable,']', '==============================')
    X = df[independent_variable]
    y = df[dependent_variable]
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and fit the Logistic Regression model
    model = LogisticRegression(max_iter=max_iters,C=regularization)
    model.fit(X_train, y_train)

    classification_score = model.score(X_train, y_train)
    print(f"Classification will makes correct predictions by: {classification_score:.2f}%")
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    if show_n_iter:
        print(f"Number of Iterations: {model.n_iter_}")
    # Evaluate the model
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Display the confusion matrix
    print(f"Classification Confusion Matrix:\n {cm}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")
    # Print classification report
    # print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    print('========================================================================================================================\n')
    return model


    def support_vector_classification(df,independent_variable,dependent_variable,Regularization=1.0,Kernel='rbf',poly_degree=3,multi_class='ovr'):
        print('\nperform support_vector_classification')
        print('\n==========================','Support Vector Classification [',independent_variable,'] with [',dependent_variable,']', '==============================')
        X = df[independent_variable]
        y = df[dependent_variable]
        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create and fit the Logistic Regression model
        model = SVC(C=Regularization,kernel=Kernel,degree=poly_degree,decision_function_shape=multi_class)
        model.fit(X_train, y_train)
    
        classification_score = model.score(X_train, y_train)
        print(f"Classification will makes correct predictions by: {classification_score:.2f}%")
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Evaluate the test model
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Display the confusion matrix
        print(f"Classification Test Confusion Matrix:\n {cm}")
        accuracy_test = accuracy_score(y_test, y_pred)
        print(f"Classification Test Accuracy: {accuracy_test:.2f}")
        # Print classification report
        # print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        print('========================================================================================================================\n')
        return model


def label_encoder(df,variable,show_label_codes=True):
    import pandas as pdd
    labels = df[variable].unique()
    label_codes = pdd.Series(pdd.factorize(labels)[0], index=labels)
    df[f'Encoded {variable}'] = df[variable].map(label_codes)
    if show_label_codes:
        print(label_codes,'\n')
        print(df[f'Encoded {variable}'])


def standardization(df,variable):
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the data
    df[variable] = scaler.fit_transform(df[variable])


# In One-Way ANOVA, there is one independent variable(factor) and three or more groups (levels).
def one_way_anova(df,independent_variable,dependent_variable,kolmogorov_smirnov=True,shapiro_wilk=False,robust_anova=False,normal_anova=False):
    print('\nperform one_way_anova')
    print('\n===============================','[',independent_variable.name,'] with [',dependent_variable.name,']', '==================================')
    if normal_anova:
        # Perform one-way ANOVA using OLS method
        model = ols(f'Q("{dependent_variable.name}") ~ Q("{independent_variable.name}")', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("ANOVA Results:")
        print(anova_table)

    if robust_anova:
        # Fit robust ANOVA model (using 'hc3' robust covariance estimator)
        model = ols(f'Q("{dependent_variable.name}") ~ Q("{independent_variable.name}")', data=df).fit(
            cov_type='HC3')
        anova_table = anova_lm(model)
        # Print robust ANOVA table
        print("Robust ANOVA Table:")
        print(anova_table)

    p_value_anova = anova_table['PR(>F)'][0]  # Extract p-value from ANOVA table
    if (p_value_anova < 0.05):
        if (p_value_anova < 0.01):
            print(f'There are statistically significant differences in the mean [{dependent_variable.name}] across different categories of [{independent_variable.name}] with confidence level = 99%.')
        else:
            print(f'There are statistically significant differences in the mean [{dependent_variable.name}] across different categories of [{independent_variable.name}] with confidence level = 95%.')
    else:
        print(f'There are [no] statistically significant differences in the mean [{dependent_variable.name}] across different categories of [{independent_variable.name}].')

    if normal_anova:
        # Perform Tukey's HSD test
        mc = MultiComparison(dependent_variable, independent_variable)
        tukey_results = mc.tukeyhsd()
        print("\nTukey's HSD Results:")
        print(tukey_results)
    if robust_anova:
        # Perform Dunn's test using scikit-posthocs
        dunn_results = posthoc_dunn(df, val_col=f'{dependent_variable.name}', group_col=f'{independent_variable.name}')
        print('-----------------------------------------------------------------------')
        print("Dunn's test Results:")
        print(dunn_results<0.05)
        print('-----------------------------------------------------------------------')


    # perform Kolmogorov-Smirnov test for normality if N>2000
    if kolmogorov_smirnov:
        test_stat_normality, p_value_normality = scipy.stats.kstest(model.resid, 'norm')
        print("p_value_normality:", p_value_normality)
        if p_value_normality < 0.05:
            print("Reject null hypothesis >> The data is not normally distributed")
        else:
            print("Fail to reject null hypothesis >> The data is normally distributed")

    # Shapiro-Wilk test for normality
    if shapiro_wilk:
        test_stat_normality, p_value_normality = scipy.stats.shapiro(model.resid)
        print("p_value_normality:", p_value_normality)
        if p_value_normality < 0.05:
            print("Reject null hypothesis >> The data is not normally distributed")
        else:
            print("Fail to reject null hypothesis >> The data is normally distributed")

    # Levene's test for homogeneity of variances
    test_stat_var, p_value_var = scipy.stats.levene(*[dependent_variable[independent_variable == group] for group in independent_variable.unique()])
    print("p_value_levene:", p_value_var)
    if p_value_var < 0.05:
        print(
            "Reject null hypothesis >> The variances of the samples are different. Non Homogeneous.use only nonparametric")
    else:
        print(
            "Fail to reject null hypothesis >> The variances of the samples are same. Homogeneous.use parametric or nonparametric")
    print('========================================================================\n')


def insert_string_at_index(df,variable,string,at_index,show_results=True):
    # Adding the string at the specified index
    df[variable] = df[variable].str.slice_replace(at_index, at_index, string)
    if show_results:
        print(df[variable])


def apply_replacements_to_specified_column_based_on_condition_in_another_column(df,column_to_update, condition_column, comparison_operator, comparison_value,new_value,show_results=True):
    comparison_operators = {"==": operator.eq,"!=": operator.ne,">": operator.gt,"<": operator.lt,">=": operator.ge,"<=": operator.le}
    comparison_func = comparison_operators.get(comparison_operator)
    if comparison_func:
        condition = comparison_func(df[condition_column], comparison_value)
        df[column_to_update].where(~condition, other=new_value, inplace=True)
        if show_results:
            print(df[column_to_update])
    else:
        print("Invalid comparison operator")


def apply_replacements_to_specified_column_based_on_regex_in_another_column(df,column_to_update, condition_column,pattern,new_value,show_results=True):
    df[column_to_update].where(~df[condition_column].str.contains(re.compile(pattern)), other=new_value,inplace=True)
    if show_results:
        print(df[column_to_update])


def replace_chars_or_pattern_within_strings_based_on_regex(df,column_to_update,pattern,new_value,show_results=True):
    df[column_to_update].replace(to_replace=pattern, value=new_value, inplace=True, regex=True)
    if show_results:
        print(df[column_to_update])


def knn_classifier(df,independent_variable,dependent_variable,K=5,majority_or_weighted_vote='uniform',euclidean_distance_or_else=2,complexity='auto'):
    print('\nperform K-Nearest Neighbors classification')
    print('\n==========================','K-Nearest Neighbors classification [',independent_variable,'] with [',dependent_variable,']', '==============================')
    X = df[independent_variable]
    y = df[dependent_variable]
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and fit the knn_classifier model
    model = KNeighborsClassifier(n_neighbors=K,weights=majority_or_weighted_vote,algorithm=complexity,p=euclidean_distance_or_else)
    model.fit(X_train, y_train)

    classification_score = model.score(X_train, y_train)
    print(f"Classification will makes correct predictions by: {classification_score:.2f}%")
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the test model
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Display the confusion matrix
    print(f"Classification Test Confusion Matrix:\n {cm}")
    accuracy_test = accuracy_score(y_test, y_pred)
    print(f"Classification Test Accuracy: {accuracy_test:.2f}")
    # Print classification report
    # print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    print('========================================================================================================================\n')
    return model


def k_means_clustering(df, features, max_clusters=10, random_state=42):
    print("Perform K-Means clustering and conduct Silhouette Analysis")
    print('========================================================================================================================')
    X = df[features]
    # Perform Silhouette Analysis to find the optimal number of clusters
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    # Determine the optimal number of clusters
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 due to 0-based indexing
    # Create and fit the K-Means model with the optimal number of clusters
    optimal_model = KMeans(n_clusters=optimal_num_clusters, n_init=10, random_state=random_state)
    cluster_labels = optimal_model.fit_predict(X)
    # Add the cluster labels to the original DataFrame
    df['Cluster'] = cluster_labels
    # Print information about the clustering
    print('\nK-Means Clustering Results:')
    print('===========================')
    print(f'Optimal Number of Clusters: {optimal_num_clusters}')
    print('========================================================================================================================\n')
    return optimal_model, cluster_labels

