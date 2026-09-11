"""
=======================================================================
PHÂN LOẠI RỦI RO TÍN DỤNG ĐA LỚP – PIPELINE HOÀN CHỈNH
Dataset: Kaggle Credit Card Approval Prediction
  - application_record.csv
  - credit_record.csv

Chạy: python credit_risk_pipeline.py
Yêu cầu: pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
=======================================================================
"""

# ───────────────────────────────────────────────
# 0. IMPORT
# ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, cohen_kappa_score, accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

import shap

# Tạo thư mục lưu kết quả
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("CREDIT RISK MULTICLASS CLASSIFICATION PIPELINE")
print("=" * 60)

# ───────────────────────────────────────────────
# 1. LOAD DATA
# ───────────────────────────────────────────────
print("\n[1/7] Loading data...")

# ĐỔI ĐƯỜNG DẪN NẾU CẦN
app = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print(f"  application_record: {app.shape}")
print(f"  credit_record:      {credit.shape}")

# ───────────────────────────────────────────────
# 2. TẠO NHÃN 4 LỚP TỪ CREDIT_RECORD
# Dựa trên logic Thông tư 11/2021/TT-NHNN NHNN
# STATUS mapping:
#   C = paid off (tất toán)
#   X = no loan in that month
#   0 = 1-29 days overdue
#   1 = 30-59 days overdue
#   2 = 60-89 days overdue
#   3 = 90-119 days overdue
#   4 = 120-149 days overdue
#   5 = overdue or bad debts, write-offs
# ───────────────────────────────────────────────
print("\n[2/7] Generating 4-class risk labels...")

STATUS_MAP = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
credit['status_num'] = credit['STATUS'].map(STATUS_MAP).fillna(0).astype(int)

# Tổng hợp theo từng ID khách hàng
def compute_features(group):
    s = group['status_num']
    return pd.Series({
        'max_status':     s.max(),
        'mean_status':    s.mean(),
        'months_bad':     (s >= 3).sum(),        # số tháng nợ xấu (nhóm 3+)
        'months_overdue': (s >= 1).sum(),        # tổng tháng có quá hạn
        'prop_paid':      (s == 0).mean(),       # tỷ lệ tháng trả đúng hạn
        'total_months':   len(s),
    })

credit_agg = credit.groupby('ID').apply(compute_features, include_groups=False).reset_index()

# Gán nhãn rủi ro 4 lớp
def assign_label(row):
    ms = row['max_status']
    mb = row['months_bad']
    if ms <= 0:
        return 0   # Thấp: luôn trả đúng hạn
    elif ms <= 2:
        return 1   # Trung bình: trễ hạn ≤ 60 ngày
    elif ms <= 4 and mb <= 2:
        return 2   # Cao: từng trễ 60-120 ngày, ít lần
    else:
        return 3   # Rất cao: nợ xấu hoặc trễ hạn nghiêm trọng

credit_agg['RISK_LABEL'] = credit_agg.apply(assign_label, axis=1)

label_names = {0: "Thấp", 1: "Trung bình", 2: "Cao", 3: "Rất cao"}
dist = credit_agg['RISK_LABEL'].value_counts().sort_index()
print("\n  Phân phối nhãn rủi ro:")
for k, v in dist.items():
    pct = v / len(credit_agg) * 100
    print(f"    Lớp {k} ({label_names[k]}): {v:,} ({pct:.1f}%)")

# ───────────────────────────────────────────────
# 3. FEATURE ENGINEERING TỪ APPLICATION_RECORD
# ───────────────────────────────────────────────
print("\n[3/7] Feature engineering...")

# Xử lý features
app_fe = app.copy()

# Chuyển ngày thành năm dương tính
app_fe['AGE'] = (-app_fe['DAYS_BIRTH'] / 365).astype(int)
app_fe['YEARS_EMPLOYED'] = app_fe['DAYS_EMPLOYED'].apply(
    lambda x: -x / 365 if x < 0 else 0  # DAYS_EMPLOYED > 0 = pensioner/unemployed
)

# Flag thất nghiệp
app_fe['IS_UNEMPLOYED'] = (app_fe['DAYS_EMPLOYED'] > 0).astype(int)

# Income per family member
app_fe['INCOME_PER_PERSON'] = app_fe['AMT_INCOME_TOTAL'] / (app_fe['CNT_FAM_MEMBERS'] + 1)

# Income per child
app_fe['INCOME_PER_CHILD'] = app_fe['AMT_INCOME_TOTAL'] / (app_fe['CNT_CHILDREN'] + 1)

# Chọn features
CATEGORICAL_COLS = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE'
]
NUMERIC_COLS = [
    'AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED',
    'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    'INCOME_PER_PERSON', 'INCOME_PER_CHILD',
    'IS_UNEMPLOYED'
]

# Encode categorical
le = LabelEncoder()
for col in CATEGORICAL_COLS:
    app_fe[col] = app_fe[col].fillna('Unknown')
    app_fe[col + '_enc'] = le.fit_transform(app_fe[col].astype(str))

ENC_COLS = [c + '_enc' for c in CATEGORICAL_COLS]
FEATURE_COLS = NUMERIC_COLS + ENC_COLS

# ───────────────────────────────────────────────
# 4. JOIN APPLICATION + CREDIT_AGG
# ───────────────────────────────────────────────
# Thêm credit features vào features của applicant
CREDIT_FEATURE_COLS = ['max_status', 'mean_status', 'months_bad',
                        'months_overdue', 'prop_paid', 'total_months']

df = app_fe[['ID'] + FEATURE_COLS].merge(
    credit_agg[['ID', 'RISK_LABEL']],
    on='ID', how='inner'
)

# Chỉ dùng application features — credit features (max_status, months_bad, ...)
# chỉ dùng để tạo nhãn, không đưa vào X (tránh data leakage)
ALL_FEATURES = FEATURE_COLS

print(f"  Dataset sau join: {df.shape[0]:,} records, {len(ALL_FEATURES)} features")

# ───────────────────────────────────────────────
# 5. TRAIN/TEST SPLIT + SMOTE-TOMEK
# ───────────────────────────────────────────────
print("\n[4/7] Splitting & resampling...")

X = df[ALL_FEATURES].fillna(0)
y = df['RISK_LABEL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Scale numeric features (cần cho LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SMOTETomek chỉ trên train set
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
print(f"  Sau SMOTETomek: {X_train_res.shape[0]:,} records")

X_train_res_scaled = scaler.fit_transform(X_train_res)

# ───────────────────────────────────────────────
# 6. TRAIN MODELS & EVALUATE
# ───────────────────────────────────────────────
print("\n[5/7] Training models...")

# Hàm tính Quadratic Weighted Kappa
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Hàm in kết quả
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, resampled=True):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    acc   = accuracy_score(y_te, y_pred)
    macro = f1_score(y_te, y_pred, average='macro')
    wf1   = f1_score(y_te, y_pred, average='weighted')
    kappa = qwk(y_te, y_pred)
    
    tag = "[resampled]" if resampled else "[original]"
    print(f"\n  {'─'*50}")
    print(f"  {name} {tag}")
    print(f"  {'─'*50}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro F1:          {macro:.4f}")
    print(f"  Weighted F1:       {wf1:.4f}")
    print(f"  QWK:               {kappa:.4f}")
    print(f"\n  Per-class F1:")
    cr = classification_report(y_te, y_pred,
                                target_names=[label_names[i] for i in range(4)],
                                output_dict=True)
    for cls in [label_names[i] for i in range(4)]:
        print(f"    {cls:12s}: P={cr[cls]['precision']:.3f}  R={cr[cls]['recall']:.3f}  F1={cr[cls]['f1-score']:.3f}")
    
    return {
        'model': name, 'resampled': resampled,
        'accuracy': acc, 'macro_f1': macro,
        'weighted_f1': wf1, 'qwk': kappa,
        'y_pred': y_pred, 'fitted_model': model
    }

results = []

# 6.1 Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
r = evaluate_model("Logistic Regression", lr,
                   X_train_res_scaled, y_train_res,
                   X_test_scaled, y_test)
results.append(r)

# 6.2 Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             random_state=42, n_jobs=-1)
r = evaluate_model("Random Forest", rf,
                   X_train_res, y_train_res,
                   X_test, y_test)
results.append(r)

# 6.3 XGBoost (standard multiclass)
label_counts = np.bincount(y_train_res)
scale_weights = len(y_train_res) / (4 * label_counts)

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42, n_jobs=-1, verbosity=0
)
r = evaluate_model("XGBoost (Multiclass)", xgb,
                   X_train_res, y_train_res,
                   X_test, y_test)
results.append(r)
best_xgb_result = r

# 6.4 LightGBM
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    num_leaves=63, class_weight='balanced',
    random_state=42, n_jobs=-1, verbose=-1
)
r = evaluate_model("LightGBM", lgbm,
                   X_train_res, y_train_res,
                   X_test, y_test)
results.append(r)

# 6.5 XGBoost với Ordinal Encoding (Frank-Hall approach)
# Biến bài toán 4 lớp thành 3 bài toán nhị phân:
#   B1: P(Y >= 1)  →  label: 0 vs {1,2,3}
#   B2: P(Y >= 2)  →  label: {0,1} vs {2,3}
#   B3: P(Y >= 3)  →  label: {0,1,2} vs {3}
print("\n  Training Ordinal XGBoost (Frank-Hall decomposition)...")

def train_ordinal_binary(X_tr, y_tr, threshold, **xgb_params):
    y_bin = (y_tr >= threshold).astype(int)
    m = XGBClassifier(**xgb_params)
    m.fit(X_tr, y_bin)
    return m

xgb_params = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0
)

m1 = train_ordinal_binary(X_train_res, y_train_res, 1, **xgb_params)
m2 = train_ordinal_binary(X_train_res, y_train_res, 2, **xgb_params)
m3 = train_ordinal_binary(X_train_res, y_train_res, 3, **xgb_params)

# Kết hợp xác suất → nhãn thứ tự
def ordinal_predict(X, m1, m2, m3):
    p1 = m1.predict_proba(X)[:, 1]  # P(Y >= 1)
    p2 = m2.predict_proba(X)[:, 1]  # P(Y >= 2)
    p3 = m3.predict_proba(X)[:, 1]  # P(Y >= 3)
    
    # P(Y = k) = P(Y >= k) - P(Y >= k+1)
    p_0 = 1 - p1
    p_1 = p1 - p2
    p_2 = p2 - p3
    p_3 = p3
    
    probs = np.stack([p_0, p_1, p_2, p_3], axis=1)
    return np.argmax(probs, axis=1)

y_pred_ordinal = ordinal_predict(X_test, m1, m2, m3)

acc   = accuracy_score(y_test, y_pred_ordinal)
macro = f1_score(y_test, y_pred_ordinal, average='macro')
wf1   = f1_score(y_test, y_pred_ordinal, average='weighted')
kappa = qwk(y_test, y_pred_ordinal)

print(f"\n  {'─'*50}")
print(f"  XGBoost Ordinal (Frank-Hall) [resampled]")
print(f"  {'─'*50}")
print(f"  Accuracy:          {acc:.4f}")
print(f"  Macro F1:          {macro:.4f}")
print(f"  Weighted F1:       {wf1:.4f}")
print(f"  QWK:               {kappa:.4f}")
print(f"\n  Per-class F1:")
cr = classification_report(y_test, y_pred_ordinal,
                            target_names=[label_names[i] for i in range(4)],
                            output_dict=True)
for cls in [label_names[i] for i in range(4)]:
    print(f"    {cls:12s}: P={cr[cls]['precision']:.3f}  R={cr[cls]['recall']:.3f}  F1={cr[cls]['f1-score']:.3f}")

results.append({
    'model': 'XGBoost Ordinal', 'resampled': True,
    'accuracy': acc, 'macro_f1': macro,
    'weighted_f1': wf1, 'qwk': kappa,
    'y_pred': y_pred_ordinal
})

# ───────────────────────────────────────────────
# 7. BẢNG SO SÁNH TẤT CẢ MODELS
# ───────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH")
print("=" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'MacroF1':>10} {'WF1':>10} {'QWK':>10}")
print("-" * 62)
for r in results:
    print(f"{r['model']:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f} {r['qwk']:>10.4f}")

# Lưu bảng so sánh tổng thể
df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'fitted_model']}
                             for r in results])
df_results.to_csv("results/model_comparison.csv", index=False)
print("\n  [Đã lưu: results/model_comparison.csv]")

# Lưu per-class F1
per_class_rows = []
for r in results:
    if 'fitted_model' not in r and r['model'] == 'XGBoost Ordinal':
        cr = classification_report(y_test, r['y_pred'],
                                   target_names=[label_names[i] for i in range(4)],
                                   output_dict=True, zero_division=0)
    elif 'fitted_model' in r:
        cr = classification_report(y_test, r['y_pred'],
                                   target_names=[label_names[i] for i in range(4)],
                                   output_dict=True, zero_division=0)
    else:
        continue
    row = {'model': r['model']}
    for i in range(4):
        row[f'f1_class{i}'] = cr[label_names[i]]['f1-score']
    per_class_rows.append(row)
pd.DataFrame(per_class_rows).to_csv("results/per_class_f1.csv", index=False)
print("  [Đã lưu: results/per_class_f1.csv]")

# ───────────────────────────────────────────────
# 8. CONFUSION MATRIX (XGBoost – best model)
# ───────────────────────────────────────────────
print("\n[6/7] Generating confusion matrix & SHAP plots...")

# Tìm model tốt nhất (QWK cao nhất)
best_result = max(results, key=lambda x: x['qwk'])
print(f"\n  Best model: {best_result['model']} (QWK = {best_result['qwk']:.4f})")

# Confusion Matrix: Random Forest (best) vs XGBoost Multiclass
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, r in zip(axes, [best_result,
                          best_xgb_result]):
    cm = confusion_matrix(y_test, r['y_pred'], normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[label_names[i] for i in range(4)],
                yticklabels=[label_names[i] for i in range(4)],
                ax=ax)
    ax.set_title(f"Confusion Matrix\n{r['model']} (QWK={r['qwk']:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [Đã lưu: results/confusion_matrix.png]")

# ───────────────────────────────────────────────
# 9. SHAP ANALYSIS (trên XGBoost Multiclass)
# ───────────────────────────────────────────────
print("\n  Computing SHAP values (có thể mất 1-2 phút)...")

# Dùng XGBoost fitted model
xgb_model = best_xgb_result['fitted_model']

# Lấy mẫu nhỏ hơn cho tốc độ
X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)

explainer  = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# Chuẩn hoá: SHAP mới trả list of arrays, cũ trả 3D array
if isinstance(shap_values, list):
    shap_3d = np.stack(shap_values, axis=2)   # → (n_samples, n_features, n_classes)
else:
    shap_3d = shap_values                      # đã là 3D array

feature_names = ALL_FEATURES

# 9.1 SHAP Summary Plot (Global – mỗi lớp 1 plot)
for class_idx, class_name in enumerate([label_names[i] for i in range(4)]):
    sv = shap_3d[:, :, class_idx]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title(f"SHAP – Lớp {class_name}")
    plt.tight_layout()
    plt.savefig(f"results/shap_class_{class_idx}_{class_name.replace(' ','_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

print("  [Đã lưu: results/shap_class_*.png]")

# 9.2 Feature Importance tổng hợp (top 15)
# Mean |SHAP| qua tất cả classes
mean_shap = np.mean([np.abs(shap_3d[:, :, c]) for c in range(4)], axis=0).mean(axis=0)
feat_imp = pd.Series(mean_shap, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(9, 6))
feat_imp.head(15).plot(kind='barh', color='steelblue')
plt.gca().invert_yaxis()
plt.title("Top 15 Features – Mean |SHAP| (all classes)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("results/shap_global_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [Đã lưu: results/shap_global_importance.png]")
print(f"\n  Top 10 features quan trọng nhất:")
for i, (feat, val) in enumerate(feat_imp.head(10).items(), 1):
    print(f"    {i:2d}. {feat:<30} {val:.4f}")

feat_imp.to_csv("results/feature_importance_shap.csv")

# ───────────────────────────────────────────────
# 10. FAIRNESS CHECK (giới tính & tuổi)
# ───────────────────────────────────────────────
print("\n[7/7] Fairness analysis...")

df_test = X_test.copy()
df_test['y_true'] = y_test.values
df_test['y_pred'] = best_result['y_pred']  # dùng best model (RF)

# 10.1 Theo giới tính (CODE_GENDER_enc)
print("\n  QWK theo giới tính (CODE_GENDER_enc):")
for g, name in [(df_test['CODE_GENDER_enc'].unique()[0], 'Nhóm A'),
                (df_test['CODE_GENDER_enc'].unique()[1], 'Nhóm B')]:
    mask = df_test['CODE_GENDER_enc'] == g
    sub = df_test[mask]
    if len(sub) > 10:
        kp = qwk(sub['y_true'], sub['y_pred'])
        mf1 = f1_score(sub['y_true'], sub['y_pred'], average='macro')
        print(f"    CODE_GENDER_enc={int(g):2d} (n={len(sub):,}): QWK={kp:.4f}, MacroF1={mf1:.4f}")

# 10.2 Theo nhóm tuổi
df_test['age_group'] = pd.cut(df_test['AGE'],
                               bins=[0, 30, 40, 50, 60, 100],
                               labels=['<30', '30-40', '40-50', '50-60', '60+'])
print("\n  QWK theo nhóm tuổi:")
for grp in ['<30', '30-40', '40-50', '50-60', '60+']:
    mask = df_test['age_group'] == grp
    sub = df_test[mask]
    if len(sub) > 10:
        kp = qwk(sub['y_true'], sub['y_pred'])
        mf1 = f1_score(sub['y_true'], sub['y_pred'], average='macro')
        print(f"    {grp:6s} (n={len(sub):,}): QWK={kp:.4f}, MacroF1={mf1:.4f}")

# Lưu fairness data
df_test[['CODE_GENDER_enc', 'AGE', 'age_group', 'y_true', 'y_pred']]\
    .to_csv("results/fairness_data.csv", index=False)
print("\n  [Đã lưu: results/fairness_data.csv]")

# ───────────────────────────────────────────────
# 11. TÓM TẮT CUỐI
# ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE HOÀN THÀNH – TÓM TẮT KẾT QUẢ")
print("=" * 60)
print(f"\n  Best model: {best_result['model']}")
print(f"  QWK:        {best_result['qwk']:.4f}")
print(f"  Macro F1:   {best_result['macro_f1']:.4f}")
print(f"  Accuracy:   {best_result['accuracy']:.4f}")
print("\n  Files đã lưu trong thư mục results/:")
print("    - model_comparison.csv      → bảng so sánh tất cả models")
print("    - confusion_matrix.png      → confusion matrix")
print("    - shap_class_*.png          → SHAP per-class plots")
print("    - shap_global_importance.png → top 15 features")
print("    - feature_importance_shap.csv → feature importance full")
print("    - fairness_data.csv         → fairness analysis data")