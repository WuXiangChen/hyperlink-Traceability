# 导入文件包
import os
import sys
import pandas as pd
import scipy.io

def readCsv(csv_path):
    return pd.read_csv(csv_path)

def readMat(mat_path):
    # 读取 .mat 文件
    return scipy.io.loadmat(mat_path)['AUC_of_Reaction_Prediction']


if __name__ == '__main__':
    root_result = "saved_results/"
    run_types = os.listdir(root_result)
    all_result = []
    for type_ in run_types:
        type_root_path = f"{root_result}/{type_}"
        csv_files = os.listdir(type_root_path)
        csv_files = [csv for csv in csv_files if csv.endswith(".csv")]

        pro_result = {"reponame":[], "CHESHIRE_avg_auc":[], "CHESHIRE_avg_recall":[], "CHESHIRE_avg_precision":[]}
        for csv_file in csv_files:
            filename = csv_file.split(".")[0]
            method_type = csv_file.split(".")[1].split("_")[0]
            pro_result["reponame"].append(filename)
            df = readCsv(f"{type_root_path}/{filename}.csv")
            auc = df["eval_auc"].mean()
            recall = df["eval_recall"].mean()
            precision = df["eval_precision"].mean()
            pro_result["CHESHIRE_avg_auc"].append(auc)
            pro_result["CHESHIRE_avg_recall"].append(recall)
            pro_result["CHESHIRE_avg_precision"].append(precision)

        pro_result["reponame"].append("avg")
        pro_result["CHESHIRE_avg_auc"].append(sum(pro_result["CHESHIRE_avg_auc"]) / len(pro_result["CHESHIRE_avg_auc"]))
        pro_result["CHESHIRE_avg_recall"].append(sum(pro_result["CHESHIRE_avg_recall"]) / len(pro_result["CHESHIRE_avg_recall"]))
        pro_result["CHESHIRE_avg_precision"].append(sum(pro_result["CHESHIRE_avg_precision"]) / len(pro_result["CHESHIRE_avg_precision"]))

        all_result.append(pd.DataFrame(pro_result))
    combined_result = pd.concat(all_result, axis=1, ignore_index=True)
    combined_result = combined_result.rename(columns={
        0: 'project',
        1: '',
        2: f'{run_types[0]}',
        3: '',
        # 4: 'project',
        # 5: '',
        # 6: f'{run_types[1]}',
        # 7: '',
        # 8: 'project',
        # 9: '',
        # 10: f'{run_types[2]}',
        # 11: '',
    })
    print(combined_result.columns)
    # 我希望在header中加入run_type
    combined_result.to_csv(f"{root_result}/all_results.csv", index=False)