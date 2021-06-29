from flask import Flask, jsonify, request
import joblib
from lightgbm import LGBMClassifier
import shap
import numpy as np
import pandas as pd
import urllib3
import pickle

app = Flask(__name__)

testX = pd.read_csv("https://raw.githubusercontent.com/VGagnot/OC_Proj_7/main/dashboard_app/echantillon_test_X.csv")
index = testX['Unnamed: 0'].copy()
testX = testX.drop(testX.columns[0], axis = 1)
testy = pd.read_csv("https://raw.githubusercontent.com/VGagnot/OC_Proj_7/main/dashboard_app/echantillon_test_y.csv")
lgbm2 = joblib.load(open("app/classification_credit.sav", 'rb'))

explainerModel = shap.TreeExplainer(lgbm2)
shap_values = explainerModel.shap_values(testX)



@app.route('/')
def shap_loc_val():
	i = int(request.args.get('individu'))
	pred = lgbm2.predict_proba(testX.iloc[i].to_numpy().reshape(1, -1))[0][1]
	top_10_contrib_idx = np.argsort(abs(shap_values[1][i]))[-10:]
	liste_contribs = []
	val_contribs = []
	col = []
	col_glob = []


	for j in top_10_contrib_idx:
		liste_contribs.append(testX.columns[j])
		val_contribs.append(shap_values[1][i][j])
		if shap_values[1][i][j] > 0:
			col.append('red')
		else:
			col.append('green')

	contrib_top_10_glob = [
		shap_values[1][i][testX.columns.tolist().index('NEW_EXT_SOURCES_SUM_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('EXT_SOURCE_2_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('EXT_SOURCE_3_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('NEW_SOURCES_PROD_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('INSTAL_DPD_MEAN_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('NEW_CREDIT_TO_GOODS_RATIO_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('NEW_CREDIT_TO_ANNUITY_RATIO_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('CODE_GENDER_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('NEW_DOC_IND_KURT_stdscl')],
		shap_values[1][i][testX.columns.tolist().index('AMT_ANNUITY_stdscl')]
		]

	for j in contrib_top_10_glob:
		if j > 0:
			col_glob.append('red')
		else:
			col_glob.append('green')


	lim_x = [
		max(map(abs, val_contribs))*1.1,
		max(map(abs, contrib_top_10_glob))*1.1,
		max(map(abs, contrib_top_10_glob + val_contribs))*1.1
		]

	control = testX['NEW_SOURCES_PROD_stdscl'].iloc[i]



	return jsonify({
		'pred':pred,
		'liste_top_10_contribs': liste_contribs,
		'val_top_10_contribs': val_contribs,
		'col': col,
		'col_glob': col_glob,
		'contrib_top_10_glob': contrib_top_10_glob,
		'lim_x': lim_x,
		'individu': i,
		'control': control})


#if __name__ == "__main__":
#    app.run(debug=False)