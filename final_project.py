import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import statsmodels.api as sm
import seaborn
import warnings

# Supress Future Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Seaborn for plots
seaborn.set()


# Returns a particular subjects data from the dataframe
def get_subject_data(data, subject_name):
    return data.loc[data['subject'] == subject_name]


# Get each distinct subject name from the dataframe
def get_subject_name(data):
    return data['subject'].drop_duplicates().tolist()


# Get each distinct age from dataframe
def get_age(data):
    return data['age'].drop_duplicates().tolist()


# Returns a particular age from the dataframe
def get_age_data(data, age):
    return data.loc[data['age'] == age]


# Convert a date to timestamp
def to_timestamp(date):
    return date.timestamp()


def main(in_directory):
    # Get data from the .csv file
    walking_data = pd.read_csv(in_directory)

    # Format the date
    walking_data['date'] = pd.to_datetime(walking_data['date'], format='%Y-%m-%d')

    # Get average walking speed
    walking_data['avg_walking_speed'] = (walking_data['min_walking_speed_kph'] + walking_data['max_walking_speed_kph']) / 2

    # Group the walking data and get only the steps per day
    grouped_walking_data = walking_data.groupby(['subject', 'date']).sum()
    steps_aggregate = grouped_walking_data.aggregate('steps').reset_index()
    pivot_steps_aggregate = steps_aggregate.pivot(index='subject', columns='date', values='steps')
    print('Number of steps taken throughout the month of March:')
    print(pivot_steps_aggregate, '\n')

    # Group data by age
    group_age_walking_data = walking_data.groupby(['age', 'date']).mean()
    age_step_aggregate = group_age_walking_data.aggregate('steps').reset_index()

    # Get age list
    ages = get_age(walking_data)

    # Get each age's data from dataframe
    age0_data = get_age_data(age_step_aggregate, ages[0])
    age1_data = get_age_data(age_step_aggregate, ages[1])
    age2_data = get_age_data(age_step_aggregate, ages[2])
    age_avg_step_list = []
    age0_step_mean = age0_data['steps'].mean()
    age_avg_step_list.append(age0_step_mean)
    age1_step_mean = age1_data['steps'].mean()
    age_avg_step_list.append(age1_step_mean)
    age2_step_mean = age2_data['steps'].mean()
    age_avg_step_list.append(age2_step_mean)

    # Get subject name list
    subjects = get_subject_name(walking_data)

    # Get each subject's data from the dataframe
    subject0_walking_data = get_subject_data(walking_data, subjects[0])
    subject1_walking_data = get_subject_data(walking_data, subjects[1])
    subject2_walking_data = get_subject_data(walking_data, subjects[2])
    subject3_walking_data = get_subject_data(walking_data, subjects[3])
    subject4_walking_data = get_subject_data(walking_data, subjects[4])
    subject5_walking_data = get_subject_data(walking_data, subjects[5])
    subject6_walking_data = get_subject_data(walking_data, subjects[6])

    # Lowess smoothing
    lowess = sm.nonparametric.lowess
    ls_subject0 = lowess(subject0_walking_data['steps'], subject0_walking_data['date'], frac=0.12)
    ls_subject1 = lowess(subject1_walking_data['steps'], subject1_walking_data['date'], frac=0.12)
    ls_subject2 = lowess(subject2_walking_data['steps'], subject2_walking_data['date'], frac=0.12)
    ls_subject3 = lowess(subject3_walking_data['steps'], subject3_walking_data['date'], frac=0.12)
    ls_subject4 = lowess(subject4_walking_data['steps'], subject4_walking_data['date'], frac=0.12)
    ls_subject5 = lowess(subject5_walking_data['steps'], subject5_walking_data['date'], frac=0.12)
    ls_subject6 = lowess(subject6_walking_data['steps'], subject6_walking_data['date'], frac=0.12)

    # Average data statistics
    subject0_drop_date = subject0_walking_data.drop('date', axis=1)
    subject0_average = subject0_drop_date.mean(axis=0)

    subject1_drop_date = subject1_walking_data.drop(['date'], axis=1)
    subject1_average = subject1_drop_date.mean(axis=0)

    subject2_drop_date = subject2_walking_data.drop(['date'], axis=1)
    subject2_average = subject2_drop_date.mean(axis=0)

    subject3_drop_date = subject3_walking_data.drop(['date'], axis=1)
    subject3_average = subject3_drop_date.mean(axis=0)

    subject4_drop_date = subject4_walking_data.drop(['date'], axis=1)
    subject4_average = subject4_drop_date.mean(axis=0)

    subject5_drop_date = subject5_walking_data.drop(['date'], axis=1)
    subject5_average = subject5_drop_date.mean(axis=0)

    subject6_drop_date = subject6_walking_data.drop(['date'], axis=1)
    subject6_average = subject6_drop_date.mean(axis=0)

    avg_data_statistics = pd.DataFrame(columns=subjects)
    avg_data_statistics[subjects[0]] = subject0_average
    avg_data_statistics[subjects[1]] = subject1_average
    avg_data_statistics[subjects[2]] = subject2_average
    avg_data_statistics[subjects[3]] = subject3_average
    avg_data_statistics[subjects[4]] = subject4_average
    avg_data_statistics[subjects[5]] = subject5_average
    avg_data_statistics[subjects[6]] = subject6_average
    print('Average Data Statistics:')
    print(avg_data_statistics, '\n')

    # Minimum Data statistics
    subject_0_min_data = subject0_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject0_min = subject_0_min_data.min()

    subject_1_min_data = subject1_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject1_min = subject_1_min_data.min()

    subject_2_min_data = subject2_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject2_min = subject_2_min_data.min()

    subject_3_min_data = subject3_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject3_min = subject_3_min_data.min()

    subject_4_min_data = subject4_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject4_min = subject_4_min_data.min()

    subject_5_min_data = subject5_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject5_min = subject_5_min_data.min()

    subject_6_min_data = subject6_walking_data[['min_step_length_cm', 'min_walking_speed_kph']]
    subject6_min = subject_6_min_data.min()

    min_data_statistics = pd.DataFrame(columns=subjects)
    min_data_statistics[subjects[0]] = subject0_min
    min_data_statistics[subjects[1]] = subject1_min
    min_data_statistics[subjects[2]] = subject2_min
    min_data_statistics[subjects[3]] = subject3_min
    min_data_statistics[subjects[4]] = subject4_min
    min_data_statistics[subjects[5]] = subject5_min
    min_data_statistics[subjects[6]] = subject6_min
    print('Minimum Data Statistics:')
    print(min_data_statistics, '\n')

    # Maximum Data Statistics
    subject_0_max_data = subject0_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject0_max = subject_0_max_data.max()

    subject_1_max_data = subject1_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject1_max = subject_1_max_data.max()

    subject_2_max_data = subject2_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject2_max = subject_2_max_data.max()

    subject_3_max_data = subject3_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject3_max = subject_3_max_data.max()

    subject_4_max_data = subject4_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject4_max = subject_4_max_data.max()

    subject_5_max_data = subject5_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject5_max = subject_5_max_data.max()

    subject_6_max_data = subject6_walking_data[['max_step_length_cm', 'max_walking_speed_kph']]
    subject6_max = subject_6_max_data.max()

    max_data_statistics = pd.DataFrame(columns=subjects)
    max_data_statistics[subjects[0]] = subject0_max
    max_data_statistics[subjects[1]] = subject1_max
    max_data_statistics[subjects[2]] = subject2_max
    max_data_statistics[subjects[3]] = subject3_max
    max_data_statistics[subjects[4]] = subject4_max
    max_data_statistics[subjects[5]] = subject5_max
    max_data_statistics[subjects[6]] = subject6_max
    print('Maximum Data Statistics:')
    print(max_data_statistics, '\n')

    # ANOVA test
    anova_p_value_steps = stats.f_oneway(walking_data['age'], walking_data['steps']).pvalue
    print("p-value for the obtained data using ANOVA test (Age and Steps): ", anova_p_value_steps, '\n')
    if anova_p_value_steps < 0.05:
        print('As p-value < 0.05 we can say that age and number of steps are related.\n')
    else:
        print('As p-value > 0.05 we can say that age and number of steps are not related.\n')

    anova_p_value_speed = stats.f_oneway(walking_data['age'], walking_data['avg_walking_speed']).pvalue
    print("p-value for the obtained data using ANOVA test (Age and Walking speed): ", anova_p_value_speed, '\n')
    print("As no p-value is obtained so its difficult to comment on the relation between age and walking speed using "
          "p-values.")

    plt.figure(num=1, figsize=(15, 8))
    # Plot the number of steps taken by each subject before filtering the data
    plt.subplot(1, 2, 1)
    plt.xticks(rotation=25)
    plt.scatter(subject0_walking_data['date'], subject0_walking_data['steps'])
    plt.scatter(subject1_walking_data['date'], subject1_walking_data['steps'])
    plt.scatter(subject2_walking_data['date'], subject2_walking_data['steps'])
    plt.scatter(subject3_walking_data['date'], subject3_walking_data['steps'])
    plt.scatter(subject4_walking_data['date'], subject4_walking_data['steps'])
    plt.scatter(subject5_walking_data['date'], subject5_walking_data['steps'])
    plt.scatter(subject6_walking_data['date'], subject6_walking_data['steps'])

    # Lowess smoothed lines
    plt.plot(subject0_walking_data['date'], ls_subject0[:, 1], 'b-')
    plt.plot(subject1_walking_data['date'], ls_subject1[:, 1], 'r-')
    plt.plot(subject2_walking_data['date'], ls_subject2[:, 1], 'g-')
    plt.plot(subject3_walking_data['date'], ls_subject3[:, 1], 'c-')
    plt.plot(subject4_walking_data['date'], ls_subject4[:, 1], 'm-')
    plt.plot(subject5_walking_data['date'], ls_subject5[:, 1], 'y-')
    plt.plot(subject6_walking_data['date'], ls_subject6[:, 1], 'b-')

    # Scatter Plot labels
    plt.xlabel('Date')
    plt.ylabel('Number of steps')
    plt.title('Steps taken by different subjects')
    plt.legend(subjects)

    # Plot the average walking speed of each subject
    plt.subplot(1, 2, 2)
    plt.xticks(rotation=25)
    plt.plot(subject0_walking_data['date'], subject0_walking_data['avg_walking_speed'])
    plt.plot(subject1_walking_data['date'], subject1_walking_data['avg_walking_speed'])
    plt.plot(subject2_walking_data['date'], subject2_walking_data['avg_walking_speed'])
    plt.plot(subject3_walking_data['date'], subject3_walking_data['avg_walking_speed'])
    plt.plot(subject4_walking_data['date'], subject4_walking_data['avg_walking_speed'])
    plt.plot(subject5_walking_data['date'], subject5_walking_data['avg_walking_speed'])
    plt.plot(subject6_walking_data['date'], subject6_walking_data['avg_walking_speed'])

    # Line graph labels
    plt.xlabel('Date')
    plt.ylabel('Average Walking Speed')
    plt.title('Average Walking Speed Of Subjects')
    plt.legend(subjects)

    # Save figure 1 as a .png file
    plt.savefig('sample_graph(1).png')

    plt.figure(num=2, figsize=(15, 8))
    # Plot the average steps taken by various age groups
    plt.subplot(1, 2, 1)
    plt.bar(ages, age_avg_step_list)

    # Bar chart labels
    plt.xlabel('Age')
    plt.ylabel('Number of Steps')
    plt.title('Average Steps per age group')

    plt.subplot(1, 2, 2)
    # Plot the walking asymmetry of each individual
    plt.xticks(rotation=25)
    plt.plot(subject0_walking_data['date'], subject0_walking_data['walking_asymmetry'])
    plt.plot(subject1_walking_data['date'], subject1_walking_data['walking_asymmetry'])
    plt.plot(subject2_walking_data['date'], subject2_walking_data['walking_asymmetry'])
    plt.plot(subject3_walking_data['date'], subject3_walking_data['walking_asymmetry'])
    plt.plot(subject4_walking_data['date'], subject4_walking_data['walking_asymmetry'])
    plt.plot(subject5_walking_data['date'], subject5_walking_data['walking_asymmetry'])
    plt.plot(subject6_walking_data['date'], subject6_walking_data['walking_asymmetry'])

    # Line graph labels
    plt.xlabel('Date')
    plt.ylabel('Walking Asymmetry')
    plt.title('Walking Asymmetry Of Subjects')
    plt.legend(subjects)

    # Save figure 2 as a .png file
    plt.savefig('sample_graph(2).png')


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
