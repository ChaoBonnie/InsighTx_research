import shap
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score


def cross_val_score(model, X_df, y_df, data_splits, instance_weights_df):
    ## report performances from the best model
    lb = OneHotEncoder()
    lb.fit(y_df.values.reshape([-1, 1]))

    sklearn_train_roc = []
    sklearn_test_roc = []

    for train_ind, test_ind in data_splits:
        ## find the train and test data for this fold
        fold_X_train = X_df.iloc[train_ind]
        fold_y_train = y_df.iloc[train_ind]
        fold_X_test = X_df.iloc[test_ind]
        fold_y_test = y_df.iloc[test_ind]

        ## binarize the target labels
        fold_y_test_binarized = lb.transform(
            fold_y_test.values.reshape([-1, 1])
        ).toarray()
        fold_weight = instance_weights_df.iloc[train_ind]

        ## fit and predict
        model.fit(fold_X_train, fold_y_train, model__sample_weight=fold_weight)
        fold_test_preds_proba = model.predict_proba(fold_X_test)

        ## find scores for this fold
        current_sklearn_test_roc = roc_auc_score(
            fold_y_test_binarized, fold_test_preds_proba, average=None
        )
        sklearn_test_roc.append(current_sklearn_test_roc)

    ## find ROC mean and std by class
    test_roc_mean_by_class = np.mean(sklearn_test_roc, axis=0)
    test_roc_std_by_class = np.std(sklearn_test_roc, axis=0)
    for i in range(test_roc_mean_by_class.shape[0]):
        print(
            "Class {}: {:.3f} +/- ({:.3f})".format(
                i, test_roc_mean_by_class[i], test_roc_std_by_class[i]
            )
        )


def get_shaps(X_df, model, shap_save_path):
    model = copy.deepcopy(model["model"])

    mybooster = model.get_booster()
    model_bytearray = mybooster.save_raw()[4:]

    def myfun(self=None):
        return model_bytearray

    mybooster.save_raw = myfun

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # saving a bar graph containing all classes
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_df, plot_type="bar")
    fig.tight_layout()
    fig.savefig(shap_save_path + "shap_bars")
    ax.clear()
    plt.close(fig)

    # saving a scatter graph for every class
    targets = [0, 1, 2]
    for target in targets:
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[target], X_df, max_display=50)
        ax.set_title("class " + str(target))
        fig.tight_layout()
        fig.savefig(shap_save_path + "shap_class" + str(target) + "_scatter")
        ax.clear()
        plt.close(fig)

    ## save all the shap values
    shap_values_mean_abs = np.mean(np.abs(shap_values), axis=1).T
    feature_names = X_df.columns
    df_shap_values_mean_abs = pd.DataFrame(
        data=shap_values_mean_abs,
        columns=["Shap Class 1", "Shap Class 2", "Shap Class 3"],
        index=feature_names,
    )
    df_shap_values_mean_abs["Shap Sum"] = df_shap_values_mean_abs.apply(
        lambda row: row["Shap Class 1"] + row["Shap Class 2"] + row["Shap Class 3"],
        axis=1,
    )

    df_shap_values_mean_abs.sort_values("Shap Sum", ascending=False, inplace=True)
    df_shap_values_mean_abs.to_csv(shap_save_path + "shap.csv", index=True)

    print("Shap values and figures saved in the selected folder.")
