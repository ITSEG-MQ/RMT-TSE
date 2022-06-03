import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa, to_table
import krippendorff

# file_1 = "Batch_4412748_batch_results.csv"
# file_1 = "Batch_4418596_batch_results.csv"
# results_raw_1 = pd.read_csv(file_1)
# file_2 = "Batch_4411467_batch_results.csv"
# results_raw_2 = pd.read_csv(file_2)
#
# hit_id = results_raw_1["HITId"].unique()
# hit_id_2 = results_raw_2["HITId"].unique()
#
# hit_id_trans = []
#
# for i in range(len(hit_id)):
#     for j in range(3):
#         hit_id_trans.append(hit_id[i])
#
# results_raw_2["HITId"] = hit_id_trans
#
# results_raw = pd.concat([results_raw_1, results_raw_2])
#
# print(results_raw.shape)
# file_1 = "Batch_4419317_batch_results.csv"
# results_raw = pd.read_csv(file_1)
# # results_raw = results_raw.drop(results_raw[results_raw["WorkerId"] == "A30US88GBGBG0V"].index)
# # filter_worker = "A30US88GBGBG0V"
# print(results_raw.shape)
def extract(dic):
    result = []
    for i in range(23):
        real = reason = 0
        for j in range(7):
            answer = dic["real_option_Q{}_{}".format(i+1,j+1)]
            if list(answer.values())[0]:
                real = j+1
                break
        for j in range(7):
            answer = dic["reason_option_Q{}_{}".format(i+1,j+1)]
            if list(answer.values())[0]:
                reason = j+1
                break
        result.append((real,reason))
    return result

def extract_hit(data, hit_no):
    hit_data = data.iloc[hit_no, :]
    workerID = [hit_data['WorkerId'] for i in range(23)]
    workTime = [hit_data['WorkTimeInSeconds'] for i in range(23)]
    hitId = [hit_data['HITId'] for i in range(23)]
    raw_answers = eval(hit_data["Answer.taskAnswers"].replace("false","False").replace("true","True"))[0]
    answers = {}
    for dic in raw_answers.items():
        # answers[dic[0]] = dic[1][dic[0]]
        answers.update(dic[1])

    experience = broken = familiar = 0
    for i, item in enumerate(['No driving experience',
                           '1 year driving experience or less',
                           '1-3 years driving experience',
                           'More than 3 years driving experience']):
        if answers[item]:
            experience = i
            break

    for i, item in enumerate(['break_law_yes','break_law_no']):
        if answers[item]:
            broken = i
            break
    for i, item in enumerate(['familiar_law_yes','familiar_law_no']):
        if answers[item]:
            familiar = i
            break

    experience = [experience for i in range(23)]
    broken = [broken for i in range(23)]
    familiar = [familiar for i in range(23)]
    reasonable = []
    real = []
    image_groups = []
    transformations = []
    for i in range(1, 24):
        image = hit_data["Input.image_" + str(i)]
        image = image.split('/')[-1]
        image = (image, image.split(')')[1][0])
        # 'Real Option Q10 1'
        for j in range(1, 8):
            if answers['Real Option Q{} {}'.format(i, j)]:
                real.append(j)
                break

        for j in range(1, 8):
            if answers['Reason Option Q{} {}'.format(i, j)]:
                reasonable.append(j)
                break

        image_groups.append(image)

        transformation = hit_data["Input.transformation_" + str(i)]
        transformations.append(transformation)

    image_groups = np.array(image_groups)
    return workerID, workTime, hitId, experience, broken, familiar, image_groups[:, 0], image_groups[:, 1], transformations, real, reasonable


def clean_raw_data():
    workerIDs = []
    workTimes = []
    hitIds = []
    workTimes = []
    experiences = []
    brokens = []
    familiars = []
    image_names = []
    violations = []
    transformations = []
    reals = []
    reasonables = []

    for i in range(results_raw.shape[0]):
        workerID, workTime, hitId, experience, broken, familiar, image_name, violation, transformation, real, reasonable = extract_hit(results_raw, i)
        workerIDs.append(workerID)
        workTimes.append(workTime)
        hitIds.append(hitId)
        experiences.append(experience)
        brokens.append(broken)
        familiars.append(familiar)
        image_names.append(image_name)
        violations.append(violation)
        transformations.append(transformation)
        reals.append(real)
        reasonables.append(reasonable)

    workerIDs = np.concatenate(workerIDs)
    workTimes = np.concatenate(workTimes)
    hitIds = np.concatenate(hitIds)
    experiences = np.concatenate(experiences)
    brokens = np.concatenate(brokens)
    familiars = np.concatenate(familiars)
    image_names = np.concatenate(image_names)
    violations = np.concatenate(violations)
    transformations = np.concatenate(transformations)
    reals = np.concatenate(reals)
    reasonables = np.concatenate(reasonables)

    # cleaned_result = "HITId,WorkerId,experience,brokenTraffic,fimilar,WorkTimeInSeconds,image,task,violation,real,resonable\n"
    clean_csv = pd.DataFrame({"WorkerId": workerIDs,
                              "WorkTimeInSeconds": workTimes,
                              "HITId": hitIds,
                              "experience": experiences,
                              "brokenTraffic": brokens,
                              "fimilar": familiars,
                              "image": image_names,
                              "task": transformations,
                              "violation": violations,
                              "real": reals,
                              "reasonable": reasonables
                              })

    clean_csv.to_csv("cleaned_result_new_2.csv", index=False)

# clean_raw_data()
# file_1 = "cleaned_result_new_2.csv"
# result = pd.read_csv(file_1)
#
# print(result.shape)
# file_2 = "cleaned_result.csv"
# result_2 = pd.read_csv(file_2)
# print(result_2.shape)

# result = pd.concat([result_1, result_2])
# print(result.shape)

def cal_kappa():
    # group by hit
    real_kappa = []
    reasonable_kappa = []
    print(len(result["HITId"].unique()))
    hits = result["HITId"].unique()
    for hit in hits:
        hit_data = result[result["HITId"] == hit]
        worker_data = [pd.DataFrame(y) for x, y in hit_data.groupby('WorkerId')]
        real_ratings = [data['real'].values for data in worker_data]
        reasonable_ratings = [data['reasonable'].values for data in worker_data]
        real_matrix = np.zeros((len(real_ratings[0]), 7))
        reasonable_matrix = np.zeros((len(real_ratings[0]), 7))
        for i in range(len(real_ratings[0])):
            for j in range(len(real_ratings)):
                real_matrix[i, real_ratings[j][i]-1] += 1
                reasonable_matrix[i, reasonable_ratings[j][i]-1] += 1

        real_kappa.append(krippendorff.alpha(real_matrix))
        reasonable_kappa.append(krippendorff.alpha(reasonable_matrix))
        # real_kappa.append(fleiss_kappa(real_matrix))
        # reasonable_kappa.append(fleiss_kappa(reasonable_matrix))
    print(real_kappa, np.mean(real_kappa))
    print(reasonable_kappa, np.mean(reasonable_kappa))
        # print('reasonable', fleiss_kappa(reasonable_matrix, 'uniform'))

    # print()
    # # group by task
    # print(len(result["HITId"].unique()))
    # hits = result["task"].unique()
    # real_kappa = []
    # reasonable_kappa = []
    # for hit in hits:
    #     hit_data = result[result["task"] == hit]
    #     worker_data = [pd.DataFrame(y) for x, y in hit_data.groupby('image')]
    #     real_ratings = [data['real'].values for data in worker_data]
    #     reasonable_ratings = [data['reasonable'].values for data in worker_data]
    #     real_matrix = np.zeros((len(real_ratings), 7))
    #     reasonable_matrix = np.zeros((len(real_ratings), 7))
    #     for i in range(len(real_ratings)):
    #         for j in range(len(real_ratings[0])):
    #             real_matrix[i, real_ratings[i][j]-1] += 1
    #             reasonable_matrix[i, reasonable_ratings[i][j]-1] += 1
    #
    #         real_kappa.append(krippendorff.alpha(real_matrix))
    #         reasonable_kappa.append(krippendorff.alpha(reasonable_matrix))
    #
    #     print(real_kappa, np.mean(real_kappa))
    #     print(reasonable_kappa, np.mean(reasonable_kappa))
    # print()
    #
    # # group by violation
    # hits = result["violation"].unique()
    # real_kappa = []
    # reasonable_kappa = []
    # for hit in hits:
    #     hit_data = result[result["violation"] == hit]
    #     worker_data = [pd.DataFrame(y) for x, y in hit_data.groupby('image')]
    #     real_ratings = [data['real'].values for data in worker_data]
    #     reasonable_ratings = [data['reasonable'].values for data in worker_data]
    #     real_matrix = np.zeros((len(real_ratings), 7))
    #     reasonable_matrix = np.zeros((len(real_ratings), 7))
    #     for i in range(len(real_ratings)):
    #         for j in range(2):
    #             real_matrix[i, real_ratings[i][j]-1] += 1
    #             reasonable_matrix[i, reasonable_ratings[i][j]-1] += 1
    #
    #         real_kappa.append(krippendorff.alpha(real_matrix))
    #         reasonable_kappa.append(krippendorff.alpha(reasonable_matrix))
    #
    #     print(real_kappa, np.mean(real_kappa))
    #     print(reasonable_kappa, np.mean(reasonable_kappa))
# cal_kappa()

def analyze_threshold():

    rules = ["verification_rule_1_VGG16(speed)", "verification_rule_2_VGG16(speed)", "validation_rule_1_VGG16(speed)",
            "validation_rule_3_VGG16(speed)", ]
    tasks = ["Add a person at roadside", "Add a speed slow sign at roadside", "Change the driving scene to night",
            "Add a person at roadside closer to the self-driving vehicle"]

    violation_dfs = []
    mturk = pd.read_csv("cleaned_result_total.csv")

    for rule, task in zip(rules, tasks):
        pred_df = pd.read_csv(rule + ".csv")
        mturk_rule = mturk[mturk["task"] == task]

        # print(mturk_rule.head())
        # mturk_rule.to_csv("task1.csv")
        worker_data = [pd.DataFrame(y) for x, y in mturk_rule.groupby('image')]
        test_images = []
        test_pred_1 = []
        test_pred_2 = []
        real_ratings = []
        reasonable_ratings = []
        workers = []
        test_pred_1_1 = []

        for i in range(len(worker_data)):
            the_image = worker_data[i]["image"].values[0]
            test_images.append(the_image)
            original_image_name = the_image.split('_')[-1]
            # print(original_image_name)
            test_pred_1.append(pred_df[pred_df["Image name"] == original_image_name]["Original pred"].values[0])
            try:
                test_pred_2.append(pred_df[pred_df["Image name"] == original_image_name]["Transformed pred"].values[0])
            except:

                test_pred_1_1.append(
                    pred_df[pred_df["Image name"] == original_image_name]["Transformed pred_1"].values[0])
                test_pred_2.append(pred_df[pred_df["Image name"] == original_image_name]["Transformed pred_2"].values[0])

            workers.append(worker_data[i]["WorkerId"].values)
            real_ratings.append(worker_data[i]["real"].values)
            reasonable_ratings.append(worker_data[i]["reasonable"].values)
        if task == "Add a person at roadside closer to the self-driving vehicle":
            p_1 = np.array(test_pred_1_1) - np.array(test_pred_1)
            p_2 = np.array(test_pred_2) - np.array(test_pred_1_1)
            pred_1 = []
            pred_2 = []

            for i in range(len(p_1)):
                print(i, p_1[i], p_2[i])
                if p_1[i] > 0:
                    pred_1.append(test_pred_1[i])
                    pred_2.append(test_pred_1_1[i])
                else:
                    pred_1.append(test_pred_1_1[i])
                    pred_2.append(test_pred_2[i])

            result_df = pd.DataFrame({"image": test_images, "original pred": pred_1, "transformed pred": pred_2,
                                 "workers": workers, "real_ratings": real_ratings, "reasonable_ratings": reasonable_ratings})
        else:
            result_df = pd.DataFrame({"image": test_images, "original pred": test_pred_1, "transformed pred": test_pred_2,
                                 "workers": workers, "real_ratings": real_ratings, "reasonable_ratings": reasonable_ratings})

        result_df["delta"] = result_df["transformed pred"] - result_df["original pred"]
        result_df["violation"] = result_df.apply(lambda row: row.image.split(')')[-1][0], axis = 1)
        result_df["mean_real"] = result_df.apply(lambda row: np.mean(row.real_ratings), axis = 1)
        result_df["mean_reasonable"] = result_df.apply(lambda row: np.mean(row.reasonable_ratings), axis = 1)
        result_df["low_reasonable"] = result_df.apply(lambda row: sum(np.array(row.reasonable_ratings) < 4), axis = 1)
        result_df.to_csv(task + "_results.csv")

        result_violation = result_df[result_df["violation"] == "1"]
        violation_dfs.append(result_violation)

    violation_total = pd.concat(violation_dfs)
    violation_total.to_csv("violation_total_speed.csv")
        # print(test_images[0])
        # print(test_pred_1[0])
        # print(test_pred_2[0])
        # print(workers[0])
        # print(real_ratings[0])
        # print(reasonable_ratings[0])


from collections import Counter

analyze_threshold()
df = pd.read_csv("violation_total_speed.csv")
print(Counter(df["violation"].values))
print(df["mean_reasonable"].describe())

import matplotlib.pyplot as plt
rate_data =[pd.DataFrame(y) for x, y in df.groupby('low_reasonable')]
delta_reasonable = {}

for i in range(len(rate_data)):
    delta_reasonable[i] = rate_data[i]["delta"]

fig, ax = plt.subplots()
ax.boxplot(delta_reasonable.values())
ax.set_xticklabels(delta_reasonable.keys())
plt.ylabel('Different of Model predictions')
plt.xlabel('No. of participants who rate the reasonability of model prediction less than 4')
plt.show()