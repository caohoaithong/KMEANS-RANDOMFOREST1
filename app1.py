
# ============================================================
# UNG DUNG MACHINE LEARNING
# K-MEANS + RANDOM FOREST
# ============================================================


# ============================================================
# 1. IMPORT THU VIEN
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# 2. CAU HINH GIAO DIEN
# ============================================================

st.set_page_config(
    page_title="Wine Intelligence | Machine Learning",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* NEN CHUNG */
.stApp {
    background: #F1F5F9;
}

/* NOI DUNG */
.block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem;
}

/* TIEU DE */
h1 {
    color: #123B70 !important;
    font-size: 38px !important;
    font-weight: 850 !important;
}

h2 {
    color: #174A8B !important;
    font-weight: 800 !important;
}

h3 {
    color: #1E40AF !important;
    font-weight: 750 !important;
}

/* THANH BEN */
section[data-testid="stSidebar"] {
    background: #E2E8F0;
}

/* CAC THE CHI SO */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 15px rgba(15,23,42,0.07);
}

div[data-testid="stMetricLabel"] {
    color: #475569;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: #2563EB;
    font-size: 30px;
    font-weight: 850;
}

/* KHUNG BANG */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.06);
}

/* KHU VUC */
.section {
    background: linear-gradient(110deg,#123B70,#2563EB);
    color: white;
    padding: 18px 22px;
    border-radius: 14px;
    margin: 25px 0 18px 0;
    font-size: 22px;
    font-weight: 800;
    box-shadow: 0 5px 12px rgba(30,64,175,.18);
}

/* NUT */
.stButton button {
    background: linear-gradient(120deg,#2563EB,#123B70);
    color: white;
    border-radius: 12px;
    min-height: 46px;
    font-weight: 750;
    border: none;
}

.stButton button:hover {
    color: white;
    transform: translateY(-2px);
}

/* FORM */
[data-testid="stForm"] {
    background: white;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 22px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 14px;
    padding: 15px;
}

/* TAB */
button[data-baseweb="tab"] {
    font-weight: 700;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. KHAI BAO DU LIEU
# ============================================================

FEATURES = [
    "fixed acidity",
    "total sulfur dioxide",
    "pH",
    "alcohol"
]

TARGET = "quality"

st.markdown("""
# 🍷 WINE INTELLIGENCE

### HỆ THỐNG PHÂN CỤM VÀ DỰ ĐOÁN CHẤT LƯỢNG RƯỢU VANG

Ứng dụng Machine Learning sử dụng hai thuật toán:

- K-means: Phân nhóm các mẫu rượu vang.
- Random Forest: Dự đoán chất lượng rượu vang.
""")

st.caption(
    "Data Analytics | Unsupervised Learning | Supervised Learning"
)

st.divider()


# ============================================================
# 4. DOC FILE EXCEL
# ============================================================

st.markdown(
    '<div class="section">📂 01. TRUY XUẤT DỮ LIỆU</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Tải lên file Excel dữ liệu rượu vang",
    type=["xlsx", "xls"]
)

if uploaded_file is None:

    st.info(
        "Hãy tải file Excel để bắt đầu phân tích. "
        "Ứng dụng hỗ trợ file XLSX và XLS."
    )

    st.stop()


# ============================================================
# 5. TIM DONG TIEU DE VA KIEM TRA TEN COT
# ============================================================

def normalize_name(name):

    return " ".join(
        str(name).strip().lower().split()
    )


required_columns = {
    "fixed acidity",
    "total sulfur dioxide",
    "ph",
    "alcohol",
    "quality"
}

try:

    raw = pd.read_excel(
        uploaded_file,
        header=None
    )

    header_index = None

    # Tim dong chua day du ten cot
    for i in range(min(20, len(raw))):

        row_values = {
            normalize_name(v)
            for v in raw.iloc[i].dropna()
        }

        if required_columns.issubset(row_values):

            header_index = i
            break

    if header_index is None:

        st.error(
            "Không tìm thấy dòng tiêu đề chứa đủ "
            "5 cột dữ liệu bắt buộc."
        )

        st.write("20 dòng đầu của dữ liệu thô:")

        st.dataframe(
            raw.head(20),
            use_container_width=True
        )

        st.stop()

    # Doc lai file voi header dung
    uploaded_file.seek(0)

    df = pd.read_excel(
        uploaded_file,
        header=header_index
    )

    # Chuan hoa ten cot
    df.columns = [
        normalize_name(c)
        for c in df.columns
    ]

    df = df.rename(columns={
        "ph": "pH"
    })

    # Kiem tra cot
    required = FEATURES + [TARGET]

    missing_columns = [
        c for c in required
        if c not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Thiếu cột dữ liệu: {missing_columns}"
        )

        st.write(
            "Các cột đang có:",
            df.columns.tolist()
        )

        st.stop()

    st.success(
        f"Đọc dữ liệu thành công! "
        f"Dòng tiêu đề được xác định tại dòng {header_index + 1}."
    )

except Exception as e:

    st.error(f"Lỗi khi đọc file Excel: {e}")

    st.stop()


# ============================================================
# 6. LAM SACH DU LIEU
# ============================================================

st.markdown(
    '<div class="section">📋 02. KIỂM TRA VÀ LÀM SẠCH DỮ LIỆU</div>',
    unsafe_allow_html=True
)

# Chuyen cac cot thanh so
for col in FEATURES + [TARGET]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Loai bo gia tri vo han
df[FEATURES + [TARGET]] = df[
    FEATURES + [TARGET]
].replace(
    [np.inf, -np.inf],
    np.nan
)

# Loai bo dong khong co nhan quality
before = len(df)

df = df.dropna(
    subset=[TARGET]
).copy()

# Quality phai la so nguyen
df = df[
    df[TARGET] % 1 == 0
].copy()

removed_rows = before - len(df)

if len(df) < 4:

    st.error(
        "Dữ liệu hợp lệ không đủ để xây dựng mô hình."
    )

    st.stop()

X = df[FEATURES].copy()

y = df[TARGET].astype(int).copy()

# Kiem tra du lieu
c1, c2, c3, c4 = st.columns(4)

c1.metric("Tổng số mẫu", len(df))

c2.metric("Số thuộc tính", len(FEATURES))

c3.metric("Số lớp quality", y.nunique())

c4.metric("Dòng bị loại", removed_rows)

st.write("### 10 dòng dữ liệu đầu tiên")

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)

tab1, tab2, tab3 = st.tabs([
    "Dữ liệu thiếu",
    "Thống kê mô tả",
    "Phân bố quality"
])

with tab1:

    missing = df[FEATURES + [TARGET]].isnull().sum()

    missing_df = missing.rename(
        "Số lượng thiếu"
    ).to_frame()

    st.dataframe(missing_df)

with tab2:

    st.dataframe(
        df[FEATURES + [TARGET]].describe().round(3),
        use_container_width=True
    )

with tab3:

    quality_count = y.value_counts().sort_index()

    fig = px.bar(
        x=quality_count.index.astype(str),
        y=quality_count.values,
        text=quality_count.values,
        labels={
            "x": "Quality",
            "y": "Số lượng mẫu"
        },
        title="PHÂN BỐ CHẤT LƯỢNG RƯỢU VANG",
        color=quality_count.values,
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 7. TIEN XU LY DU LIEU CHO K-MEANS
# ============================================================

st.markdown(
    '<div class="section">⚙️ 03. TIỀN XỬ LÝ DỮ LIỆU</div>',
    unsafe_allow_html=True
)

# K-means:
# Imputer va Scaler duoc fit tren toan bo tap X
# vi muc dich la phan cum mo ta toan bo bo du lieu.

kmeans_imputer = SimpleImputer(
    strategy="median"
)

X_kmeans_imputed = kmeans_imputer.fit_transform(X)

kmeans_scaler = StandardScaler()

X_scaled = kmeans_scaler.fit_transform(
    X_kmeans_imputed
)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=FEATURES,
    index=X.index
)

st.success(
    "Đã xử lý dữ liệu thiếu và chuẩn hóa 4 thuộc tính."
)

st.write("### Dữ liệu sau chuẩn hóa")

st.dataframe(
    X_scaled_df.head(10).round(3),
    use_container_width=True
)


# ============================================================
# 8. ELBOW METHOD
# ============================================================

st.markdown(
    '<div class="section">📈 04. XÁC ĐỊNH SỐ CỤM BẰNG ELBOW METHOD</div>',
    unsafe_allow_html=True
)

max_k = min(20, len(X_scaled) - 1)

inertias = []

with st.spinner("Đang tính toán Elbow Method..."):

    for k in range(1, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
            max_iter=300
        )

        model.fit(X_scaled)

        inertias.append(
            model.inertia_
        )

elbow_df = pd.DataFrame({
    "K": range(1, max_k + 1),
    "Inertia": inertias
})

fig = px.line(
    elbow_df,
    x="K",
    y="Inertia",
    markers=True,
    title="ELBOW METHOD - K = 1 ĐẾN 20"
)

fig.update_layout(
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write("### Bảng giá trị Inertia")

st.dataframe(
    elbow_df.round(4),
    use_container_width=True,
    hide_index=True
)

st.info(
    "Điểm Elbow là vùng mà tốc độ giảm Inertia bắt đầu chậm lại. "
    "Ứng dụng không tự khẳng định một K tối ưu chỉ từ biểu đồ."
)


# ============================================================
# 9. LUA CHON K VA HUAN LUYEN K-MEANS
# ============================================================

st.markdown(
    '<div class="section">🔵 05. PHÂN CỤM K-MEANS</div>',
    unsafe_allow_html=True
)

k_selected = st.slider(
    "Chọn số cụm K",
    min_value=2,
    max_value=max_k,
    value=min(3, max_k),
    step=1
)

kmeans = KMeans(
    n_clusters=k_selected,
    random_state=42,
    n_init=20,
    max_iter=300
)

clusters = kmeans.fit_predict(
    X_scaled
)

df_result = df.copy()

df_result["Cluster"] = clusters

cluster_count = (
    df_result["Cluster"]
    .value_counts()
    .sort_index()
)

c1, c2, c3 = st.columns(3)

c1.metric("Số cụm", k_selected)

c2.metric("Số vòng lặp", kmeans.n_iter_)

c3.metric("Inertia", f"{kmeans.inertia_:.4f}")

st.write("### Bảng kết quả phân cụm")

st.dataframe(
    df_result,
    use_container_width=True,
    hide_index=True
)

st.write("### Phân bố số lượng mẫu")

fig = px.bar(
    x=cluster_count.index.astype(str),
    y=cluster_count.values,
    text=cluster_count.values,
    labels={
        "x": "Cluster",
        "y": "Số lượng mẫu"
    },
    title="SỐ LƯỢNG MẪU TRONG TỪNG CỤM",
    color=cluster_count.values,
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 10. PHAN TICH DAC DIEM CAC CUM
# ============================================================

st.markdown(
    '<div class="section">📊 06. PHÂN TÍCH ĐẶC ĐIỂM CÁC CỤM</div>',
    unsafe_allow_html=True
)

cluster_mean = df_result.groupby(
    "Cluster"
)[FEATURES + [TARGET]].mean()

cluster_std = df_result.groupby(
    "Cluster"
)[FEATURES + [TARGET]].std()

cluster_count = df_result.groupby(
    "Cluster"
).size()

cluster_summary = cluster_mean.copy()

cluster_summary.insert(
    0,
    "Số lượng mẫu",
    cluster_count
)

cluster_summary.insert(
    1,
    "Tỷ lệ (%)",
    cluster_count / len(df_result) * 100
)

st.write("### Bảng tổng hợp đặc điểm từng cụm")

st.dataframe(
    cluster_summary.round(3),
    use_container_width=True
)

st.write("### So sánh đặc điểm hóa học")

fig = go.Figure()

for feature in FEATURES:

    fig.add_trace(
        go.Bar(
            name=feature,
            x=cluster_mean.index.astype(str),
            y=cluster_mean[feature]
        )
    )

fig.update_layout(
    title="GIÁ TRỊ TRUNG BÌNH CÁC THUỘC TÍNH THEO CỤM",
    xaxis_title="Cluster",
    yaxis_title="Giá trị trung bình",
    barmode="group",
    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Nhan xet tung cum
st.write("### Giải thích đặc điểm từng cụm")

for cluster_id in cluster_mean.index:

    values = cluster_mean.loc[cluster_id]

    with st.expander(
        f"🔹 CỤM {cluster_id}",
        expanded=True
    ):

        st.write(
            f"**Số lượng mẫu:** {cluster_count[cluster_id]}"
        )

        st.write(
            f"**Tỷ lệ:** "
            f"{cluster_count[cluster_id] / len(df_result) * 100:.2f}%"
        )

        for feature in FEATURES:

            st.write(
                f"- {feature}: {values[feature]:.3f}"
            )

        st.write(
            f"- Quality trung bình: {values[TARGET]:.3f}"
        )

        st.caption(
            "Nhận xét: Đây là nhóm mẫu có đặc điểm hóa học "
            "được thể hiện qua các giá trị trung bình phía trên. "
            "Tên cụm chỉ mang tính định danh, không phải thứ hạng chất lượng."
        )


# ============================================================
# 11. DANH GIA K-MEANS
# ============================================================

st.markdown(
    '<div class="section">📐 07. ĐÁNH GIÁ MÔ HÌNH K-MEANS</div>',
    unsafe_allow_html=True
)

if 1 < len(np.unique(clusters)) < len(X_scaled):

    sil = silhouette_score(
        X_scaled,
        clusters
    )

    db = davies_bouldin_score(
        X_scaled,
        clusters
    )

    ch = calinski_harabasz_score(
        X_scaled,
        clusters
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Silhouette Score",
        f"{sil:.4f}"
    )

    metric_col2.metric(
        "Davies-Bouldin",
        f"{db:.4f}"
    )

    metric_col3.metric(
        "Calinski-Harabasz",
        f"{ch:.4f}"
    )

    st.write("""
    **Cách đọc kết quả:**

    - Silhouette Score: Càng gần 1 càng tốt.
    - Davies-Bouldin: Càng thấp càng tốt.
    - Calinski-Harabasz: Thường càng cao càng tốt.

    Cần xem xét đồng thời các chỉ số và đặc điểm thực tế của cụm.
    """)

else:

    st.warning(
        "Không đủ điều kiện để tính các chỉ số đánh giá cụm."
    )


# ============================================================
# 12. TRUC QUAN HOA PCA
# ============================================================

st.markdown(
    '<div class="section">🎨 08. TRỰC QUAN HÓA PHÂN CỤM PCA</div>',
    unsafe_allow_html=True
)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame({
    "PCA 1": X_pca[:, 0],
    "PCA 2": X_pca[:, 1],
    "Cluster": clusters.astype(str),
    "Quality": y.values
})

fig = px.scatter(
    pca_df,
    x="PCA 1",
    y="PCA 2",
    color="Cluster",
    symbol="Cluster",
    hover_data=["Quality"],
    title="PHÂN CỤM K-MEANS TRÊN KHÔNG GIAN PCA",
    color_discrete_sequence=px.colors.qualitative.Bold
)

centers_pca = pca.transform(
    kmeans.cluster_centers_
)

fig.add_trace(
    go.Scatter(
        x=centers_pca[:, 0],
        y=centers_pca[:, 1],
        mode="markers",
        marker=dict(
            symbol="x",
            size=18,
            color="black",
            line=dict(width=3)
        ),
        name="Tâm cụm"
    )
)

fig.update_layout(
    template="plotly_white",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

explained = sum(
    pca.explained_variance_ratio_
) * 100

st.info(
    f"Hai thành phần PCA giải thích {explained:.2f}% "
    "phương sai của dữ liệu."
)


# ============================================================
# 13. RANDOM FOREST - CROSS VALIDATION
# ============================================================

st.markdown(
    '<div class="section">🌳 09. HUẤN LUYỆN RANDOM FOREST</div>',
    unsafe_allow_html=True
)

class_counts = y.value_counts()

min_class = int(class_counts.min())

if min_class < 2:

    st.error(
        "Có lớp quality chỉ có một mẫu. "
        "Không đủ dữ liệu để thực hiện Stratified Cross Validation."
    )

    st.stop()

n_splits = min(
    4,
    min_class
)

n_repeats = 5

cv = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=42
)


# Tao mo hinh
def create_rf_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])


st.info(
    f"Phương pháp: {n_splits}-Fold Cross Validation, "
    f"lặp lại {n_repeats} lần."
)

st.write("""
Mỗi mẫu được dự đoán khi nó nằm trong tập kiểm tra,
không tham gia huấn luyện ở lượt tương ứng.
""")

classes = np.sort(
    y.unique()
)

prob_sum = np.zeros(
    (len(X), len(classes))
)

prediction_count = np.zeros(
    len(X)
)

fold_results = []

progress = st.progress(0)

total_folds = n_splits * n_repeats

with st.spinner("Đang huấn luyện và đánh giá Random Forest..."):

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]

        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        y_test = y.iloc[test_idx]

        model = create_rf_model()

        model.fit(
            X_train,
            y_train
        )

        fold_pred = model.predict(
            X_test
        )

        fold_prob = model.predict_proba(
            X_test
        )

        model_classes = (
            model.named_steps["forest"].classes_
        )

        positions = np.searchsorted(
            classes,
            model_classes
        )

        prob_sum[
            np.ix_(test_idx, positions)
        ] += fold_prob

        prediction_count[test_idx] += 1

        fold_results.append({

            "Fold": fold_number,

            "Accuracy": accuracy_score(
                y_test,
                fold_pred
            ),

            "Precision": precision_score(
                y_test,
                fold_pred,
                average="macro",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                fold_pred,
                average="macro",
                zero_division=0
            ),

            "F1-score": f1_score(
                y_test,
                fold_pred,
                average="macro",
                zero_division=0
            )

        })

        progress.progress(
            fold_number / total_folds
        )


# ============================================================
# 14. TONG HOP DU DOAN TOAN BO MAU
# ============================================================

if np.any(prediction_count == 0):

    st.error(
        "Có mẫu chưa được đánh giá. "
        "Không thể tiếp tục."
    )

    st.stop()

average_prob = (
    prob_sum / prediction_count[:, None]
)

y_pred = classes[
    np.argmax(
        average_prob,
        axis=1
    )
]


# ============================================================
# 15. TINH CAC CHI SO DANH GIA
# ============================================================

accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y,
    y_pred
)


# ============================================================
# 16. SO SANH BASELINE
# ============================================================

baseline_pred = np.zeros(
    len(y),
    dtype=int
)

majority_class = y.mode().iloc[0]

baseline_pred[:] = majority_class

baseline_accuracy = accuracy_score(
    y,
    baseline_pred
)


# ============================================================
# 17. HIEN THI KET QUA RANDOM FOREST
# ============================================================

st.markdown(
    '<div class="section">📊 10. ĐÁNH GIÁ TỔNG THỂ RANDOM FOREST</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Accuracy",
    f"{accuracy:.3f}"
)

c2.metric(
    "Precision",
    f"{precision:.3f}"
)

c3.metric(
    "Recall",
    f"{recall:.3f}"
)

c4.metric(
    "F1-score",
    f"{f1:.3f}"
)

c5.metric(
    "Balanced Accuracy",
    f"{balanced_acc:.3f}"
)

st.write("### So sánh mô hình cơ sở")

comparison = pd.DataFrame({

    "Mô hình": [
        "Majority Baseline",
        "Random Forest"
    ],

    "Accuracy": [
        baseline_accuracy,
        accuracy
    ]

})

fig = px.bar(
    comparison,
    x="Mô hình",
    y="Accuracy",
    color="Mô hình",
    text_auto=".3f",
    title="SO SÁNH ACCURACY"
)

fig.update_layout(
    template="plotly_white",
    yaxis_range=[0, 1]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 18. BANG DU DOAN CHUNG
# ============================================================

st.write("### Bảng dự đoán chung trên toàn bộ dữ liệu")

evaluation = X.copy()

evaluation["Quality thực tế"] = y.values

evaluation["Quality dự đoán"] = y_pred

evaluation["Kết quả"] = np.where(
    y.values == y_pred,
    "Đúng",
    "Sai"
)

evaluation["Độ tin cậy (%)"] = (
    average_prob.max(axis=1) * 100
).round(2)

st.dataframe(
    evaluation,
    use_container_width=True,
    hide_index=True
)

correct = int(
    np.sum(y.values == y_pred)
)

incorrect = len(y) - correct

c1, c2, c3 = st.columns(3)

c1.metric(
    "Dự đoán đúng",
    correct
)

c2.metric(
    "Dự đoán sai",
    incorrect
)

c3.metric(
    "Tổng số mẫu",
    len(y)
)

# Tai file ket qua
csv_data = evaluation.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Tải bảng dự đoán CSV",
    data=csv_data,
    file_name="ket_qua_random_forest.csv",
    mime="text/csv"
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

st.write("### Ma trận nhầm lẫn")

cm = confusion_matrix(
    y,
    y_pred,
    labels=classes
)

fig = px.imshow(
    cm,
    text_auto=True,
    x=classes.astype(str),
    y=classes.astype(str),
    color_continuous_scale="Blues",
    labels={
        "x": "Quality dự đoán",
        "y": "Quality thực tế",
        "color": "Số mẫu"
    },
    title="CONFUSION MATRIX - RANDOM FOREST"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 20. DANH GIA QUA CAC FOLD
# ============================================================

st.write("### Đánh giá độ ổn định qua các Fold")

fold_df = pd.DataFrame(
    fold_results
)

st.dataframe(
    fold_df.round(4),
    use_container_width=True,
    hide_index=True
)

fold_summary = fold_df.drop(
    columns="Fold"
).agg([
    "mean",
    "std"
]).T

fold_summary.columns = [
    "Trung bình",
    "Độ lệch chuẩn"
]

st.write("### Trung bình và độ lệch chuẩn")

st.dataframe(
    fold_summary.round(4),
    use_container_width=True
)

fig = px.box(
    fold_df.melt(
        id_vars="Fold",
        var_name="Chỉ số",
        value_name="Giá trị"
    ),
    x="Chỉ số",
    y="Giá trị",
    color="Chỉ số",
    title="ĐỘ ỔN ĐỊNH CÁC CHỈ SỐ QUA CROSS VALIDATION"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section">🌲 11. MỨC ĐỘ QUAN TRỌNG CỦA THUỘC TÍNH</div>',
    unsafe_allow_html=True
)

# Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu
# Chỉ dùng mô hình này cho dự đoán dữ liệu mới

rf_model = create_rf_model()

rf_model.fit(
    X,
    y
)

importance = (
    rf_model.named_steps["forest"]
    .feature_importances_
)

importance_df = pd.DataFrame({

    "Thuộc tính": FEATURES,

    "Mức độ quan trọng": importance

}).sort_values(
    "Mức độ quan trọng",
    ascending=False
)

fig = px.bar(
    importance_df,
    x="Mức độ quan trọng",
    y="Thuộc tính",
    orientation="h",
    text_auto=".4f",
    color="Mức độ quan trọng",
    color_continuous_scale="Blues",
    title="FEATURE IMPORTANCE - RANDOM FOREST"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 22. DU DOAN DU LIEU MOI
# ============================================================

st.markdown(
    '<div class="section">🔮 12. DỰ ĐOÁN DỮ LIỆU MỚI</div>',
    unsafe_allow_html=True
)

st.write("""
Nhập các thuộc tính hóa học để hệ thống dự đoán:

- Nhóm cụm bằng K-means.
- Chất lượng rượu bằng Random Forest.
- Xác suất dự đoán của từng lớp quality.
""")

medians = X.median()

mins = X.min()

maxs = X.max()

with st.form("prediction_form"):

    st.write("### Nhập thông số hóa học")

    col1, col2 = st.columns(2)

    with col1:

        new_acidity = st.number_input(
            "Fixed acidity",
            value=float(medians["fixed acidity"]),
            format="%.4f"
        )

        new_sulfur = st.number_input(
            "Total sulfur dioxide",
            value=float(medians["total sulfur dioxide"]),
            format="%.4f"
        )

    with col2:

        new_ph = st.number_input(
            "pH",
            value=float(medians["pH"]),
            format="%.4f"
        )

        new_alcohol = st.number_input(
            "Alcohol",
            value=float(medians["alcohol"]),
            format="%.4f"
        )

    submitted = st.form_submit_button(
        "🔍 THỰC HIỆN DỰ ĐOÁN",
        use_container_width=True
    )


if submitted:

    new_data = pd.DataFrame({

        "fixed acidity": [new_acidity],

        "total sulfur dioxide": [new_sulfur],

        "pH": [new_ph],

        "alcohol": [new_alcohol]

    })

    # K-means
    new_imputed = kmeans_imputer.transform(
        new_data
    )

    new_scaled = kmeans_scaler.transform(
        new_imputed
    )

    new_cluster = kmeans.predict(
        new_scaled
    )[0]

    # Random Forest
    new_quality = rf_model.predict(
        new_data
    )[0]

    new_probability = rf_model.predict_proba(
        new_data
    )[0]

    confidence = new_probability.max() * 100

    st.success(
        "Hoàn tất dự đoán!"
    )

    st.write("### Kết quả dự đoán")

    a, b, c = st.columns(3)

    a.metric(
        "K-means Cluster",
        f"Cluster {new_cluster}"
    )

    b.metric(
        "Random Forest Quality",
        str(new_quality)
    )

    c.metric(
        "Xác suất lớp cao nhất",
        f"{confidence:.2f}%"
    )

    st.write("### Dữ liệu đầu vào")

    st.dataframe(
        new_data,
        use_container_width=True
    )

    st.write("### Xác suất dự đoán các lớp")

    probability_df = pd.DataFrame({

        "Quality": rf_model.classes_.astype(str),

        "Xác suất (%)": new_probability * 100

    })

    fig = px.bar(
        probability_df,
        x="Quality",
        y="Xác suất (%)",
        color="Xác suất (%)",
        text_auto=".2f",
        title="XÁC SUẤT DỰ ĐOÁN RANDOM FOREST",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Kiem tra ngoai pham vi du lieu
    out_of_range = []

    for feature in FEATURES:

        if (
            new_data.iloc[0][feature] < mins[feature]
            or new_data.iloc[0][feature] > maxs[feature]
        ):

            out_of_range.append(feature)

    if out_of_range:

        st.warning(
            "Một số thông số nằm ngoài phạm vi dữ liệu "
            f"huấn luyện: {', '.join(out_of_range)}. "
            "Kết quả dự đoán có thể kém tin cậy hơn."
        )

    st.info(
        "Xác suất là mức độ phân phối dự đoán của mô hình, "
        "không phải xác suất bảo đảm kết quả đúng. "
        "Mô hình cuối cùng được huấn luyện trên toàn bộ dữ liệu hiện có."
    )


# ============================================================
# 23. TONG KET
# ============================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:25px;
    background:linear-gradient(120deg,#123B70,#2563EB);
    border-radius:16px;
    color:white;
">

    <div style="font-size:24px;font-weight:850;">
        🍷 WINE INTELLIGENCE
    </div>

    <div style="font-size:15px;margin-top:8px;">
        MACHINE LEARNING ANALYTICS
    </div>

    <div style="font-size:13px;margin-top:12px;color:#DBEAFE;">
        K-MEANS | RANDOM FOREST | DATA ANALYTICS
    </div>

</div>
""", unsafe_allow_html=True)
